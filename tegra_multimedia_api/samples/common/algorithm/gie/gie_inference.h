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
#ifndef GIE_INFERENCE_H_
#define GIE_INFERENCE_H_

#include <fstream>
#include <queue>
#include "NvInfer.h"
#include "NvCaffeParser.h"
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/objdetect/objdetect.hpp>
using namespace nvinfer1;
using namespace nvcaffeparser1;
using namespace std;

// Model Index
#define GOOGLENET_SINGLE_CLASS 0
#define GOOGLENET_THREE_CLASS  1
#define HELNET_THREE_CLASS  2

struct {
    const int  classCnt;
    float      THRESHOLD[3];
    const char *INPUT_BLOB_NAME;
    const char *OUTPUT_BLOB_NAME;
    const char *OUTPUT_BBOX_NAME;
    const int  STRIDE;
    const int  WORKSPACE_SIZE;
    float      bbox_output_scales[4];
} *g_pModelNetAttr, gModelNetAttr[3] = {

    {
        // GOOGLENET_SINGLE_CLASS
        1,
        {0.8, 0, 0},
        "data",
        "coverage",
        "bboxes",
        4,
        450 * 1024 * 1024,
        {1, 1, 1, 1}
    },

    {
        // GOOGLENET_THREE_CLASS
        3,
        {0.6, 0.6, 1.0},   //People, Motorbike, Car
        "data",
        "Layer16_cov",
        "Layer16_bbox",
        16,
        110 * 1024 * 1024,
        {-640, -368, 640, 368}
    },

    {
        // HELNET_THREE_CLASS
        1,
        {0.8, 0xffff, 0xffff},
        "data",
        "Layer19_cov",
        "Layer19_bbox",
        16,
        450 * 1024 * 1024,
        {-640, -480, 640, 480}
    }
};

class Logger;

class Profiler;

class GIE_Context
{
public:
    //net related parameter
    int getNetWidth() const;

    int getNetHeight() const;

    int getBatchSize() const;

    int getChannel() const;

    int getModelClassCnt() const;

    // Buffer is allocated in GIE_Conxtex,
    // Expose this interface for inputing data
    void*& getBuffer(const int& index);

    float*& getInputBuf();

    int getNumGieInstances() const;

    void setForcedFp32(const bool& forced_fp32);

    void setDumpResult(const bool& dump_result);

    void setGieProfilerEnabled(const bool& enable_gie_profiler);

    int getFilterNum() const;
    void setFilterNum(const unsigned int& filter_num);

    GIE_Context();

    void setModelIndex(int modelIndex);

    void buildGieContext(const string& deployfile,
            const string& modelfile, bool bUseCPUBuf = false);

    void doInference(
        queue< vector<cv::Rect> >* rectList_queue,
        float *input = NULL);

    void destroyGieContext(bool bUseCPUBuf = false);

    ~GIE_Context();

private:
    int net_width;
    int net_height;
    int filter_num;
    void  **buffers;
    float *input_buf;
    float *output_cov_buf;
    float *output_bbox_buf;
    float helnet_scale[4];
    IRuntime *runtime;
    ICudaEngine *engine;
    IExecutionContext *context;
    uint32_t *pResultArray;
    int channel;              //input file's channel
    int num_bindings;
    int gieinstance_num;      //inference channel num
    int batch_size;
    bool forced_fp32;
    bool dump_result;
    ofstream fstream;
    bool enable_gie_profiler;
    stringstream gieModelStream;
    vector<string> outputs;
    string result_file;
    Logger *pLogger;
    Profiler *pProfiler;
    int frame_num;
    uint64_t elapsed_frame_num;
    uint64_t elapsed_time;
    int inputIndex;
    int outputIndex;
    int outputIndexBBOX;
    Dims3 inputDims;
    Dims3 outputDims;
    Dims3 outputDimsBBOX;
    size_t inputSize;
    size_t outputSize;
    size_t outputSizeBBOX;

    int parseNet(const string& deployfile);
    void parseBbox(vector<cv::Rect>* rectList, int batch_th);
    void allocateMemory(bool bUseCPUBuf);
    void releaseMemory(bool bUseCPUBuf);
    void caffeToGIEModel(const string& deployfile, const string& modelfile);
};

#endif
