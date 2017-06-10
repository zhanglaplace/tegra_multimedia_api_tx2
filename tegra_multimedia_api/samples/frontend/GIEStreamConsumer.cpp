/*
 * Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <fstream>
#include <iostream>
#include <assert.h>
#include <opencv2/core/core.hpp>
#include "GIEStreamConsumer.h"
#include <EGLStream/NV/ImageNativeBuffer.h>
#include "Error.h"
#include "NvCudaProc.h"
#include "NvEglRenderer.h"

#define TIMESPEC_DIFF_USEC(timespec1, timespec2) \
    (((timespec1)->tv_sec - (timespec2)->tv_sec) * 1000000L + \
        (timespec1)->tv_usec - (timespec2)->tv_usec)

#define IS_EOS_BUFFER(buf)  (buf.fd < 0)
#define MAX_QUEUE_SIZE      (10)
#define MAX_GIE_BUFFER      (10)
#define GIE_INTERVAL        (1)

#define GIE_MODEL    GOOGLENET_THREE_CLASS

extern bool g_bVerbose;
extern bool g_bProfiling;

GIEStreamConsumer::GIEStreamConsumer(const char *name, const char *outputFilename,
        Size2D<uint32_t> size, NvEglRenderer *renderer, bool hasEncoding) :
    StreamConsumer(name, size),
    m_VideoEncoder(name, outputFilename, size.width(), size.height(), V4L2_PIX_FMT_H264),
    m_hasEncoding(hasEncoding),
    m_eglRenderer(renderer)
{
    nvosd_context = nvosd_create_context();
    m_VideoEncoder.setBufferDoneCallback(bufferDoneCallback, this);
}

GIEStreamConsumer::~GIEStreamConsumer()
{
    nvosd_destroy_context(nvosd_context);
}

bool GIEStreamConsumer::threadInitialize()
{
    if (!StreamConsumer::threadInitialize())
        return false;

    // Init encoder
    if (m_hasEncoding)
        m_VideoEncoder.initialize();

    // Create GIE model
    Log("Creating GIE model..\n");
    m_GIEContext.setModelIndex(GIE_MODEL);
    m_GIEContext.buildGieContext(m_deployFile, m_modelFile);
    m_GIEContext.setGieProfilerEnabled(true);
    Log("Batch size: %d\n", m_GIEContext.getBatchSize());

    // Check if we have enough buffer for batch
    if (GIE_INTERVAL * m_GIEContext.getBatchSize() > MAX_QUEUE_SIZE)
        ORIGINATE_ERROR("GIE_INTERVAL(%d) * BATCH_SIZE must less or equal to QUEUE_SIZE(%d)",
                GIE_INTERVAL, MAX_QUEUE_SIZE);

    // Create buffers
    for (unsigned i = 0; i < MAX_QUEUE_SIZE; i++)
    {
        int dmabuf_fd;

        if (NvBufferCreate(&dmabuf_fd, m_size.width(), m_size.height(),
                    NvBufferLayout_BlockLinear, NvBufferColorFormat_YUV420) < 0)
            ORIGINATE_ERROR("Failed to create NvBuffer.");

        m_emptyBufferQueue.push(dmabuf_fd);
    }

    // Create GIE buffers
    for (unsigned i = 0; i < MAX_GIE_BUFFER; i++)
    {
        int fd;
        NvBufferCreate(&fd, m_GIEContext.getNetWidth(), m_GIEContext.getNetHeight(),
                NvBufferLayout_Pitch, NvBufferColorFormat_ABGR32);
        m_emptyGIEBufferQueue.push(fd);
    }

    // Launch render and GIE threads
    pthread_create(&m_renderThread, NULL, RenderThreadProc, this);
    pthread_create(&m_gieThread, NULL, GIEThreadProc, this);

    return true;
}

bool GIEStreamConsumer::processFrame(Frame *frame)
{
    IFrame *iFrame = interface_cast<IFrame>(frame);
    if (!iFrame)
    {
        static BufferInfo eosBuffer = { -1 };   // EOS
        m_gieBufferQueue.push(eosBuffer);
        m_renderBufferQueue.push(eosBuffer);
        return false;
    }

    BufferInfo buf;
    buf.fd = m_emptyBufferQueue.pop();
    buf.number = iFrame->getNumber();

    // Get the IImageNativeBuffer extension interface and create the fd.
    NV::IImageNativeBuffer *iNativeBuffer =
        interface_cast<NV::IImageNativeBuffer>(iFrame->getImage());
    if (!iNativeBuffer)
        ORIGINATE_ERROR("IImageNativeBuffer not supported by Image.");

    iNativeBuffer->copyToNvBuffer(buf.fd);

    // Do GIE inference every 10 frames
    if (iFrame->getNumber() % GIE_INTERVAL == 0)
    {
        BufferInfo gieBuf;
        gieBuf.fd = m_emptyGIEBufferQueue.pop();
        gieBuf.number = iFrame->getNumber();
        iNativeBuffer->copyToNvBuffer(gieBuf.fd);
        m_gieBufferQueue.push(gieBuf);
    }

    m_renderBufferQueue.push(buf);

    return true;
}

bool GIEStreamConsumer::threadShutdown()
{
    pthread_join(m_renderThread, NULL);
    pthread_join(m_gieThread, NULL);

    if (m_hasEncoding)
        m_VideoEncoder.shutdown();

    // Ensure all buffers are returned by encoder
    assert(m_emptyBufferQueue.size() == MAX_QUEUE_SIZE);

    m_GIEContext.destroyGieContext();

    // Destroy all buffers
    while (m_emptyBufferQueue.size())
        NvBufferDestroy(m_emptyBufferQueue.pop());

    while (m_emptyGIEBufferQueue.size())
        NvBufferDestroy(m_emptyGIEBufferQueue.pop());

    return StreamConsumer::threadShutdown();
}

bool GIEStreamConsumer::RenderThreadProc()
{
    Log("Render thread started.\n");

    // Start profiling
    if (m_eglRenderer)
        m_eglRenderer->enableProfiling();

    while (true)
    {
        BufferInfo buf = m_renderBufferQueue.pop();

        if (!IS_EOS_BUFFER(buf))
        {
            if (buf.number % GIE_INTERVAL == 0)
            {
                m_rectParams.clear();

                // Get bound box info from GIE thread
                for (int class_num = 0; class_num < m_GIEContext.getModelClassCnt(); class_num++)
                {
                    vector<Rect2f> *bbox = m_bboxesQueue[class_num].pop();

                    if (bbox)   // bbox = NULL means GIE thread has exited
                    {
                        for (unsigned i = 0; i < bbox->size(); i++)
                        {
                            Rect2f &rect = bbox->at(i);
                            NvOSD_RectParams rectParam = { 0 };
                            rectParam.left   = m_size.width()  * rect.x;
                            rectParam.top    = m_size.height() * rect.y;
                            rectParam.width  = m_size.width()  * rect.width;
                            rectParam.height = m_size.height() * rect.height;
                            rectParam.border_width = 5;
                            rectParam.border_color.red = ((class_num == 0) ? 1.0f : 0.0);
                            rectParam.border_color.green = ((class_num == 1) ? 1.0f : 0.0);
                            rectParam.border_color.blue = ((class_num == 2) ? 1.0f : 0.0);
                            m_rectParams.push_back(rectParam);
                        }
                        delete bbox;
                    }
                }
            }

            if (g_bVerbose)
                Log("Render: processing frame %d\n", buf.number);

            // Draw bounding box
            nvosd_draw_rectangles(nvosd_context, MODE_HW, buf.fd,
                        m_rectParams.size(), m_rectParams.data());

            // Do rendering
            if (m_eglRenderer)
            {
                if (buf.number % 30 == 0)
                {
                    NvElementProfiler::NvElementProfilerData data;
                    m_eglRenderer->getProfilingData(data);

                    uint64_t framesCount = data.total_processed_units -
                        m_profilerData.total_processed_units;
                    uint64_t timeElapsed = TIMESPEC_DIFF_USEC(&data.profiling_time,
                            &m_profilerData.profiling_time);
                    m_fps = (float)framesCount * 1e6 / timeElapsed;
                    memcpy(&m_profilerData, &data, sizeof(data));
                    Log("FPS: %f\n", m_fps);
                }

                char overlay[256];
                snprintf(overlay, sizeof(overlay), "Frame %u, FPS %f", buf.number, m_fps);
                m_eglRenderer->setOverlayText(overlay, 10, 30);
                m_eglRenderer->render(buf.fd);
            }
        }

        // Do encoding
        if (m_hasEncoding)
            m_VideoEncoder.encodeFromFd(buf.fd);
        else if (!IS_EOS_BUFFER(buf))
            bufferDoneCallback(buf.fd);

        if (IS_EOS_BUFFER(buf))
            break;
    }

    // Print profiling stats
    if (m_eglRenderer)
        m_eglRenderer->printProfilingStats();

    Log("Render thread exited.\n");
    return true;
}

bool GIEStreamConsumer::GIEThreadProc()
{
    IStream *iStream = interface_cast<IStream>(m_stream);
    Log("GIE thread started.\n");

    unsigned bufNumInBatch = 0;
    int class_num = 0;
    int classCnt = m_GIEContext.getModelClassCnt();

    while (true)
    {
        BufferInfo buf = m_gieBufferQueue.pop();
        if (IS_EOS_BUFFER(buf))
            break;

        if (g_bVerbose)
            Log("GIE: Add frame %d to batch (%d/%d)\n", buf.number, bufNumInBatch,
                    m_GIEContext.getBatchSize());

        EGLImageKHR eglImage = NvEGLImageFromFd(iStream->getEGLDisplay(), buf.fd);
        size_t batchOffset = bufNumInBatch * m_GIEContext.getNetWidth() *
            m_GIEContext.getNetHeight() * m_GIEContext.getChannel();
        mapEGLImage2Float(&eglImage,
                m_GIEContext.getNetWidth(),
                m_GIEContext.getNetHeight(),
                (GIE_MODEL == GOOGLENET_THREE_CLASS) ? COLOR_FORMAT_BGR : COLOR_FORMAT_RGB,
                (char*) m_GIEContext.getBuffer(0) + batchOffset * sizeof(float));
        NvDestroyEGLImage(iStream->getEGLDisplay(), eglImage);
        m_emptyGIEBufferQueue.push(buf.fd);

        if (++bufNumInBatch < m_GIEContext.getBatchSize())
            continue;       // Batch not ready, wait for new buffers

        // Inference
        queue<vector<cv::Rect>> rectList_queue[classCnt];
        m_GIEContext.doInference(rectList_queue);

        for (int i = 0; i < classCnt; i++)
        {
            assert(rectList_queue[i].size() == m_GIEContext.getBatchSize());
        }

        for (class_num = 0; class_num < classCnt; class_num++)
        {
            for ( ; !rectList_queue[class_num].empty(); rectList_queue[class_num].pop())
            {
                vector<cv::Rect> &rectList = rectList_queue[class_num].front();
                vector<Rect2f> *bbox = new vector<Rect2f>();

                // Calculate normalized bound box
                for (vector<cv::Rect>::iterator it = rectList.begin(); it != rectList.end(); it++)
                {
                    cv::Rect rect = *it;
                    Rect2f rect2f;
                    rect2f.x      = (float)rect.x      / m_GIEContext.getNetWidth();
                    rect2f.y      = (float)rect.y      / m_GIEContext.getNetHeight();
                    rect2f.width  = (float)rect.width  / m_GIEContext.getNetWidth();
                    rect2f.height = (float)rect.height / m_GIEContext.getNetHeight();

                    bbox->push_back(rect2f);
                }
                m_bboxesQueue[class_num].push(bbox);
            }
        }

        if (g_bVerbose)
            Log("GIE: Batch done (%d/%d)\n", bufNumInBatch, m_GIEContext.getBatchSize());

        bufNumInBatch = 0;  // Reset counter for next batch
    }

    // Tell render thread we are exiting.
    for (class_num = 0; class_num < classCnt; class_num++)
    {
        for (int i = 0; i < m_GIEContext.getBatchSize(); i++)
            m_bboxesQueue[class_num].push(NULL);
    }

    Log("GIE thread exited.\n");

    return true;
}

void GIEStreamConsumer::bufferDoneCallback(int dmabuf_fd)
{
    m_emptyBufferQueue.push(dmabuf_fd);
}
