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

#include "gie_inference.h"
#include <stdlib.h>
#include <sys/time.h>
#include <assert.h>
#include <sstream>
#include <iostream>
#include <sys/stat.h>
#include <cmath>
#include <cuda_runtime_api.h>

static const int TIMING_ITERATIONS = 1;
static const int NUM_BINDINGS = 3;
static const int FILTER_NUM = 6;

#define CHECK(status)                                   \
{                                                       \
    if (status != 0)                                    \
    {                                                   \
        std::cout << "Cuda failure: " << status;        \
        abort();                                        \
    }                                                   \
}

// Logger for GIE info/warning/errors
class Logger : public ILogger
{
    void log(Severity severity, const char* msg) override
    {
        // suppress info-level messages
        if (severity != Severity::kINFO)
            std::cout << msg << std::endl;
    }
};

class Profiler : public IProfiler
{
    typedef std::pair<std::string, float> Record;
    std::vector<Record> mProfile;

    virtual void reportLayerTime(const char* layerName, float ms)
    {
        auto record = std::find_if(mProfile.begin(), mProfile.end(),
                        [&](const Record& r){ return r.first == layerName; });
        if (record == mProfile.end())
            mProfile.push_back(std::make_pair(layerName, ms));
        else
            record->second += ms;
    }

    void printLayerTimes()
    {
        float totalTime = 0;
        for (size_t i = 0; i < mProfile.size(); i++)
        {
            printf("%-40.40s %4.3fms\n", mProfile[i].first.c_str(),
                    mProfile[i].second / TIMING_ITERATIONS);
            totalTime += mProfile[i].second;
        }
        printf("Time over all layers: %4.3f\n", totalTime / TIMING_ITERATIONS);
    }

};

string stringtrim(string);

//This function is used to trim space
string
stringtrim(string s)
{
    int i = 0;
    while (s[i] == ' ')
    {
        i++;
    }
    s = s.substr(i);
    i = s.size()-1;
    while (s[i] == ' ')
    {
        i--;
    }

    s = s.substr(0, i + 1);
    return s;
}

int
GIE_Context::getNetWidth() const
{
    return net_width;
}

int
GIE_Context::getNetHeight() const
{
    return net_height;
}


int
GIE_Context::getFilterNum() const
{
    return filter_num;
}

void
GIE_Context::setFilterNum(const unsigned int& filter_num)
{
    this->filter_num = filter_num;
}

void*&
GIE_Context::getBuffer(const int& index)
{
    assert(index >= 0 && index < num_bindings);
    return buffers[index];
}

float*&
GIE_Context::getInputBuf()
{
    return input_buf;
}

int
GIE_Context::getNumGieInstances() const
{
    return gieinstance_num;
}

int
GIE_Context::getBatchSize() const
{
    return batch_size;
}

int
GIE_Context::getModelClassCnt() const
{
    return g_pModelNetAttr->classCnt;
}

void
GIE_Context::setForcedFp32(const bool& forced_fp32)
{
    this->forced_fp32 = forced_fp32;
}

void
GIE_Context::setDumpResult(const bool& dump_result)
{
    this->dump_result = dump_result;
}

void
GIE_Context::setGieProfilerEnabled(const bool& enable_gie_profiler)
{
    this->enable_gie_profiler = enable_gie_profiler;
}

int
GIE_Context::getChannel() const
{
    return channel;
}


GIE_Context::GIE_Context()
{
    net_width = 0;
    net_height = 0;
    filter_num = FILTER_NUM;
    buffers = new void *[NUM_BINDINGS];
    for (int i = 0; i < NUM_BINDINGS; i++)
    {
        buffers[i] = NULL;
    }
    input_buf = NULL;
    output_cov_buf = NULL;
    output_bbox_buf = NULL;

    runtime = NULL;
    engine = NULL;
    context = NULL;
    pResultArray = new uint32_t[100*4];

    channel = 0;
    num_bindings = NUM_BINDINGS;

    batch_size = 0;

    gieinstance_num = 1;

    forced_fp32 = 0;
    elapsed_frame_num = 0;
    elapsed_time = 0;
    enable_gie_profiler = 1;
    dump_result = 0;
    frame_num = 0;
    result_file = "result.txt";
    pLogger = new Logger;
    pProfiler = new Profiler;
}


void
GIE_Context::allocateMemory(bool bUseCPUBuf)
{
    const ICudaEngine& cuda_engine = context->getEngine();
    // input and output buffer pointers that we pass to the engine
    // the engine requires exactly IEngine::getNbBindings() of these
    // but in this case we know that there is exactly one input and one output
    assert(cuda_engine.getNbBindings() == num_bindings);

    // In order to bind the buffers, we need to know the names of the input
    // and output tensors. note that indices are guaranteed to be less than
    // IEngine::getNbBindings()
    inputIndex = cuda_engine.getBindingIndex(g_pModelNetAttr->INPUT_BLOB_NAME);
    outputIndex = cuda_engine.getBindingIndex(g_pModelNetAttr->OUTPUT_BLOB_NAME);
    outputIndexBBOX = cuda_engine.getBindingIndex(g_pModelNetAttr->OUTPUT_BBOX_NAME);
    // allocate GPU buffers
    inputDims = cuda_engine.getBindingDimensions(inputIndex);
    outputDims = cuda_engine.getBindingDimensions(outputIndex);
    outputDimsBBOX = cuda_engine.getBindingDimensions(outputIndexBBOX);

    inputSize = batch_size * inputDims.c * inputDims.h * inputDims.w *
                            sizeof(float);
    outputSize = batch_size * outputDims.c * outputDims.h *
                            outputDims.w * sizeof(float);
    printf("outputDim c %d w %d h %d\n", outputDims.c, outputDims.w, outputDims.h);
    outputSizeBBOX = batch_size * outputDimsBBOX.c * outputDimsBBOX.h *
                            outputDimsBBOX.w * sizeof(float);
    printf("outputDimsBBOX c %d w %d h %d\n", outputDimsBBOX.c, outputDimsBBOX.w, outputDimsBBOX.h);
    if (bUseCPUBuf && input_buf == NULL)
    {
        input_buf = (float *)malloc(inputSize);
        assert(input_buf != NULL);
    }

    if (output_cov_buf == NULL)
    {
        output_cov_buf = (float *)malloc(outputSize);
        assert(output_cov_buf != NULL);
    }
    if (outputIndexBBOX >= 0)
    {
        if (output_bbox_buf == NULL)
        {
            output_bbox_buf = (float *)malloc(outputSizeBBOX);
            assert(output_bbox_buf != NULL);
        }
    }
    // create GPU buffers and a stream
    if (buffers[inputIndex] == NULL)
    {
        CHECK(cudaMalloc(&buffers[inputIndex], inputSize));
    }
    if (buffers[outputIndex] == NULL)
    {
        CHECK(cudaMalloc(&buffers[outputIndex], outputSize));
    }
    if (outputIndexBBOX >= 0)
    {
        if (buffers[outputIndexBBOX] == NULL)
        {
            CHECK(cudaMalloc(&buffers[outputIndexBBOX], outputSizeBBOX));
        }
    }
    if (dump_result)
    {
        fstream.open(result_file.c_str(), ios::out);
    }
}

void
GIE_Context::releaseMemory(bool bUseCPUBuf)
{
    for (int i = 0; i < NUM_BINDINGS; i++)
    {
        if (buffers[i] != NULL)
        {
            CHECK(cudaFree(buffers[i]));
            buffers[i] = NULL;
        }
    }
    if (bUseCPUBuf && input_buf != NULL)
    {
        free(input_buf);
        input_buf = NULL;
    }
    if (output_cov_buf != NULL)
    {
        free(output_cov_buf);
        output_cov_buf = NULL;
    }
    if (output_bbox_buf != NULL)
    {
        free(output_bbox_buf);
        output_bbox_buf = NULL;
    }

    if (pResultArray != NULL)
    {
        delete []pResultArray;
        pResultArray = NULL;
    }

    if (dump_result)
    {
        fstream.close();
    }
}

GIE_Context::~GIE_Context()
{

    delete pLogger;
    delete pProfiler;
    delete []buffers;
}

void
GIE_Context::caffeToGIEModel(const string& deployfile, const string& modelfile)
{
    // create API root class - must span the lifetime of the engine usage
    IBuilder *builder = createInferBuilder(*pLogger);
    INetworkDefinition *network = builder->createNetwork();

    // parse the caffe model to populate the network, then set the outputs
    ICaffeParser *parser = createCaffeParser();

    bool hasFp16 = builder->platformHasFastFp16();

    // if user specify
    if (forced_fp32 == false)
    {
        if (hasFp16)
        {
            printf("forced_fp32 has been set to 0(using fp16)\n");
        }
        else
        {
            printf("platform don't have fp16, force to 1(using fp32)\n");
        }
    }
    else if(forced_fp32 == true)
    {
        printf("forced_fp32 has been set to 1(using fp32)\n");
        hasFp16 = 0;
    }

    // create a 16-bit model if it's natively supported
    DataType modelDataType = hasFp16 ? DataType::kHALF : DataType::kFLOAT;
    const IBlobNameToTensor *blobNameToTensor =
        parser->parse(deployfile.c_str(),    // caffe deploy file
                      modelfile.c_str(),     // caffe model file
                      *network,              // network definition that parser populate
                      modelDataType);
    assert(blobNameToTensor != nullptr);

    // the caffe file has no notion of outputs
    // so we need to manually say which tensors the engine should generate
    outputs = {g_pModelNetAttr->OUTPUT_BLOB_NAME,
               g_pModelNetAttr->OUTPUT_BBOX_NAME};
    for (auto& s : outputs)
    {
        network->markOutput(*blobNameToTensor->find(s.c_str()));
        printf("outputs %s\n", s.c_str());
    }

    // Build the engine
    builder->setMaxBatchSize(batch_size);
    builder->setMaxWorkspaceSize(g_pModelNetAttr->WORKSPACE_SIZE);

    // Eliminate the side-effect from the delay of GPU frequency boost
    builder->setMinFindIterations(3);
    builder->setAverageFindIterations(2);

    // set up the network for paired-fp16 format, only on DriveCX
    if (hasFp16)
    {
        builder->setHalf2Mode(true);
    }

    ICudaEngine *engine = builder->buildCudaEngine(*network);
    assert(engine);

    // we don't need the network any more, and we can destroy the parser
    network->destroy();
    parser->destroy();

    // serialize the engine, then close everything down
    engine->serialize(gieModelStream);
    engine->destroy();
    builder->destroy();
    shutdownProtobufLibrary();
}

void
GIE_Context::setModelIndex(int index)
{
    assert(index == GOOGLENET_SINGLE_CLASS ||
           index == GOOGLENET_THREE_CLASS ||
           index == HELNET_THREE_CLASS);
    g_pModelNetAttr = gModelNetAttr + index;
    assert(g_pModelNetAttr->classCnt > 0);
    assert(g_pModelNetAttr->STRIDE > 0);
    assert(g_pModelNetAttr->WORKSPACE_SIZE > 0);
}

void
GIE_Context::buildGieContext(const string& deployfile,
        const string& modelfile, bool bUseCPUBuf)
{
    if (!parseNet(deployfile))
    {
        cout<<"parse net failed, exit!"<<endl;
        exit(0);
    }
    ifstream gieModelFile("gieModel.cache");
    if (gieModelFile.good())
    {
        cout<<"Using cached GIE model"<<endl;
        gieModelStream << gieModelFile.rdbuf();
    }
    else
    {
        caffeToGIEModel(deployfile, modelfile);
        cout<<"Create GIE model cache"<<endl;
        ofstream gieModelFile("gieModel.cache");
        gieModelFile << gieModelStream.rdbuf();
        gieModelFile.close();
    }
    gieModelStream.seekg(0, gieModelStream.beg);
    runtime = createInferRuntime(*pLogger);
    engine = runtime->deserializeCudaEngine(gieModelStream);
    context = engine->createExecutionContext();
    allocateMemory(bUseCPUBuf);
}

void
GIE_Context::destroyGieContext(bool bUseCPUBuf)
{
    releaseMemory(bUseCPUBuf);
    context->destroy();
    engine->destroy();
    runtime->destroy();
}

void
GIE_Context::doInference(
    queue< vector<cv::Rect> >* rectList_queue,
    float *input)
{
    struct timeval input_time;
    struct timeval output_time;

    if (!enable_gie_profiler)
    {
        cudaStream_t stream;
        CHECK(cudaStreamCreate(&stream));

        // DMA the input to the GPU,  execute the batch asynchronously
        // and DMA it back
        if (input != NULL)   //NULL means we have use GPU to map memory
        {
            CHECK(cudaMemcpyAsync(buffers[inputIndex], input, inputSize,
                                cudaMemcpyHostToDevice, stream));
        }

        context->enqueue(batch_size, buffers, stream, nullptr);
        CHECK(cudaMemcpyAsync(output_cov_buf, buffers[outputIndex], outputSize,
                                cudaMemcpyDeviceToHost, stream));
        if (outputIndexBBOX >= 0)
        {
            CHECK(cudaMemcpyAsync(output_bbox_buf, buffers[outputIndexBBOX],
                            outputSizeBBOX, cudaMemcpyDeviceToHost, stream));
        }

        cudaStreamSynchronize(stream);

        // release the stream and the buffers
        cudaStreamDestroy(stream);
    }
    else
    {
        // DMA the input to the GPU,  execute the batch synchronously
        // and DMA it back
        if (input != NULL)   //NULL means we have use GPU to map memory
        {
            CHECK(cudaMemcpy(buffers[inputIndex], input, inputSize,
                                cudaMemcpyHostToDevice));
        }

        gettimeofday(&input_time, NULL);
        context->execute(batch_size, buffers);
        gettimeofday(&output_time, NULL);
        CHECK(cudaMemcpy(output_cov_buf, buffers[outputIndex], outputSize,
                                cudaMemcpyDeviceToHost));
        if (outputIndexBBOX >= 0)
        {
            CHECK(cudaMemcpy(output_bbox_buf, buffers[outputIndexBBOX],
                            outputSizeBBOX, cudaMemcpyDeviceToHost));
        }
        elapsed_frame_num += batch_size;
        elapsed_time += (output_time.tv_sec - input_time.tv_sec) * 1000 +
                        (output_time.tv_usec - input_time.tv_usec) / 1000;
        if (elapsed_frame_num >= 100)
        {
            printf("Time elapsed:%ld ms per frame in past %ld frames\n",
                elapsed_time / elapsed_frame_num, elapsed_frame_num);
            elapsed_frame_num = 0;
            elapsed_time = 0;
        }
    }

    vector<cv::Rect> rectList[getModelClassCnt()];
    for (int i = 0; i < batch_size; i++)
    {
        parseBbox(rectList, i);

        if (dump_result)
        {
            for (int class_num = 0; class_num < getModelClassCnt(); class_num++)
            {
                fstream << "frame:" << frame_num << " class num:" << class_num
                        << " has rect:" << rectList[class_num].size() << endl;
                for (int i = 0; i < rectList[class_num].size(); i++)
                {
                    fstream << "\tx,y,w,h:"
                            << (float) rectList[class_num][i].x / net_width << " "
                            << (float) rectList[class_num][i].y / net_height << " "
                            << (float) rectList[class_num][i].width / net_width << " "
                            << (float) rectList[class_num][i].height / net_height << endl;
                }
                fstream << endl;
            }
            frame_num++;
        }

        for (int class_num = 0; class_num < getModelClassCnt(); class_num++)
        {
            rectList_queue[class_num].push(rectList[class_num]);
        }
    }
}

void
GIE_Context::parseBbox(vector<cv::Rect>* rectList, int batch_th)
{
    int gridsize = outputDims.h * outputDims.w;
    int gridoffset = outputDims.c * outputDims.h * outputDims.w * batch_th;

    for (int class_num = 0; class_num < getModelClassCnt(); class_num++)
    {
        float *output_x1 = output_bbox_buf +
                outputDimsBBOX.c * outputDimsBBOX.h * outputDimsBBOX.w * batch_th +
                class_num * 4 * outputDimsBBOX.h * outputDimsBBOX.w;
        float *output_y1 = output_x1 + outputDimsBBOX.h * outputDimsBBOX.w;
        float *output_x2 = output_y1 + outputDimsBBOX.h * outputDimsBBOX.w;
        float *output_y2 = output_x2 + outputDimsBBOX.h * outputDimsBBOX.w;

        for (int i = 0; i < gridsize; ++i)
        {
            if (output_cov_buf[gridoffset + class_num * gridsize + i] >=
                                          g_pModelNetAttr->THRESHOLD[class_num])
            {
                int g_x = i % outputDims.w;
                int g_y = i / outputDims.w;
                int i_x = g_x * g_pModelNetAttr->STRIDE;
                int i_y = g_y * g_pModelNetAttr->STRIDE;
                int rectx1 = g_pModelNetAttr->bbox_output_scales[0] * output_x1[i] + i_x;
                int recty1 = g_pModelNetAttr->bbox_output_scales[1] * output_y1[i] + i_y;
                int rectx2 = g_pModelNetAttr->bbox_output_scales[2] * output_x2[i] + i_x;
                int recty2 = g_pModelNetAttr->bbox_output_scales[3] * output_y2[i] + i_y;
                if (rectx1 < 0)
                {
                    rectx1 = 0;
                }
                if (rectx2 < 0)
                {
                    rectx2 = 0;
                }
                if (recty1 < 0)
                {
                    recty1 = 0;
                }
                if (recty2 < 0)
                {
                    recty2 = 0;
                }
                if (rectx1 >= (int)net_width)
                {
                    rectx1 = net_width - 1;
                }
                if (rectx2 >= (int)net_width)
                {
                    rectx2 = net_width - 1;
                }
                if (recty1 >= (int)net_height)
                {
                    recty1 = net_height - 1;
                }
                if (recty2 >= (int)net_height)
                {
                    recty2 = net_height - 1;
                }
                rectList[class_num].push_back(cv::Rect(rectx1, recty1,
                                                      rectx2 - rectx1, recty2 - recty1));
            }
        }

        cv::groupRectangles(rectList[class_num], 3, 0.2);
    }
}

int
GIE_Context::parseNet(const string& deployfile)
{
    ifstream readfile;
    string line;
    readfile.open(deployfile, ios::in);
    if (!readfile)
    {
        return 0;
    }
    int iterator = 0;

    while (1)
    {
        getline(readfile, line);
        string::size_type index;

        index = line.find("input_dim");
        if (index ==std::string::npos)
        {
            continue;
        }

        index = line.find_first_of(":", 0);
        string first = line.substr(0, index);
        string second = line.substr(index + 1);
        switch(iterator)
        {
        case 0:  //for batch size
            batch_size = atoi(stringtrim(second).c_str());
            assert(batch_size > 0);
            break;

        case 1:  // for channel num in net
            channel = atoi(stringtrim(second).c_str());
            assert(channel > 0);
            break;

        case 2:  // for net's height
            net_height = atoi(stringtrim(second).c_str());
            assert(net_height > 0);
            break;
        case 3:  // for net's width
            net_width = atoi(stringtrim(second).c_str());
            assert(net_width > 0);
            break;

        default:
            break;
        }

        if (iterator == 3)
        {
            break;
        }

        iterator++;
    }

    cout<<"Net has batch_size, channel, net_height, net_width:" <<
        batch_size << " " << channel << " " << net_height << " " <<
        net_width << endl;

    if (net_height == 480 && net_width == 640)
    {
        setModelIndex(HELNET_THREE_CLASS);
    }

    readfile.close();

    return 1;
}
