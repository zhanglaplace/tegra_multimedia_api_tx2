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

#include "NvUtils.h"
#include <fstream>
#include <iostream>
#include <linux/videodev2.h>
#include <malloc.h>
#include <sstream>
#include <string.h>

#include "video_encode.h"

#define TEST_ERROR(cond, str, label) if(cond) { \
                                        cerr << str << endl; \
                                        error = 1; \
                                        goto label; }

#define TEST_PARSE_ERROR(cond, label) if(cond) { \
    cerr << "Error parsing runtime parameter changes string" << endl; \
    goto label; }

#define IS_DIGIT(c) (c >= '0' && c <= '9')

using namespace std;

static void
abort(context_t *ctx)
{
    ctx->got_error = true;
    ctx->enc->abort();
}

//Initialise CRC Rec and creates CRC Table based on the polynomial.
static
Crc* InitCrc(unsigned int CrcPolynomial)
{
    unsigned short int i;
    unsigned short int j;
    unsigned int tempcrc;
    Crc *phCrc;
    phCrc = (Crc*) malloc (sizeof(Crc));
    if (phCrc == NULL)
    {
        cerr << "Mem allocation failed for Init CRC" <<endl;
        return NULL;
    }

    memset (phCrc, 0, sizeof(Crc));

    for (i = 0; i <= 255; i++)
    {
        tempcrc = i;
        for (j = 8; j > 0; j--)
        {
            if (tempcrc & 1)
            {
                tempcrc = (tempcrc >> 1) ^ CrcPolynomial;
            }
            else
            {
                tempcrc >>= 1;
            }
        }
        phCrc->CRCTable[i] = tempcrc;
    }

    phCrc->CrcValue = 0;
    return phCrc;
}

//Calculates CRC of data provided in by buffer.
static
void CalculateCrc(Crc *phCrc, unsigned char *buffer, uint32_t count)
{
    unsigned char *p;
    unsigned int temp1;
    unsigned int temp2;
    unsigned int crc = phCrc->CrcValue;
    unsigned int *CRCTable = phCrc->CRCTable;

    if(!count)
        return;

    p = (unsigned char *) buffer;
    while (count-- != 0)
    {
        temp1 = (crc >> 8) & 0x00FFFFFFL;
        temp2 = CRCTable[((unsigned int) crc ^ *p++) & 0xFF];
        crc = temp1 ^ temp2;
    }

    phCrc->CrcValue = crc;
}

//Closes CRC related handles.
static
void CloseCrc(Crc **phCrc)
{
    if (*phCrc)
        free (*phCrc);
}

static int
write_encoder_output_frame(ofstream * stream, NvBuffer * buffer)
{
    stream->write((char *) buffer->planes[0].data, buffer->planes[0].bytesused);
    return 0;
}

static bool
encoder_capture_plane_dq_callback(struct v4l2_buffer *v4l2_buf, NvBuffer * buffer,
                                  NvBuffer * shared_buffer, void *arg)
{
    context_t *ctx = (context_t *) arg;
    NvVideoEncoder *enc = ctx->enc;
    uint32_t frame_num = ctx->enc->capture_plane.getTotalDequeuedBuffers() - 1;
    uint32_t ReconRef_Y_CRC = 0;
    uint32_t ReconRef_U_CRC = 0;
    uint32_t ReconRef_V_CRC = 0;

    if (v4l2_buf == NULL)
    {
        cout << "Error while dequeing buffer from output plane" << endl;
        abort(ctx);
        return false;
    }

    // GOT EOS from encoder. Stop dqthread.
    if (buffer->planes[0].bytesused == 0)
    {
        return false;
    }

    // Computing CRC with each frame
    if(ctx->pBitStreamCrc)
        CalculateCrc (ctx->pBitStreamCrc, buffer->planes[0].data, buffer->planes[0].bytesused);


    write_encoder_output_frame(ctx->out_file, buffer);

    if (ctx->report_metadata)
    {
        v4l2_ctrl_videoenc_outputbuf_metadata enc_metadata;
        if (ctx->enc->getMetadata(v4l2_buf->index, enc_metadata) == 0)
        {
            if (ctx->bReconCrc && enc_metadata.bValidReconCRC) {
                // CRC for Recon frame
                cout << "Frame: " << frame_num << endl;
                cout << "ReconFrame_Y_CRC " << enc_metadata.ReconFrame_Y_CRC <<
                    " ReconFrame_U_CRC " << enc_metadata.ReconFrame_U_CRC <<
                    " ReconFrame_V_CRC " << enc_metadata.ReconFrame_V_CRC <<
                    endl;

                if (!ctx->recon_Ref_file->eof()) {
                    *ctx->recon_Ref_file >> ReconRef_Y_CRC;
                    *ctx->recon_Ref_file >> ReconRef_U_CRC;
                    *ctx->recon_Ref_file >> ReconRef_V_CRC;
                }

                if ((ReconRef_Y_CRC != enc_metadata.ReconFrame_Y_CRC) ||
                    (ReconRef_U_CRC != enc_metadata.ReconFrame_U_CRC) ||
                    (ReconRef_V_CRC != enc_metadata.ReconFrame_V_CRC))
                {
                    cout << "Recon CRC FAIL" << endl;
                    cout << "ReconRef_Y_CRC " << ReconRef_Y_CRC <<
                        " ReconRef_U_CRC " << ReconRef_U_CRC <<
                        " ReconRef_V_CRC " << ReconRef_V_CRC <<
                        endl;
                    abort(ctx);
                    return false;
                }
                cout << "Recon CRC PASS for frame : " << frame_num << endl;
            } else if (ctx->externalRPS && enc_metadata.bRPSFeedback_status) {
                // RPS Feedback
                cout << "Frame: " << frame_num << endl;
                cout << "nCurrentRefFrameId " << enc_metadata.nCurrentRefFrameId <<
                     " nActiveRefFrames " << enc_metadata.nActiveRefFrames << endl;

                for (uint32_t i = 0; i < enc_metadata.nActiveRefFrames; i++)
                {
                    cout << "FrameId " << enc_metadata.RPSList[i].nFrameId <<
                     " IdrFrame " <<  (int) enc_metadata.RPSList[i].bIdrFrame <<
                     " LTRefFrame " <<  (int) enc_metadata.RPSList[i].bLTRefFrame <<
                     " PictureOrderCnt " << enc_metadata.RPSList[i].nPictureOrderCnt <<
                     " FrameNum " << enc_metadata.RPSList[i].nFrameNum <<
                     " LTFrameIdx " <<  enc_metadata.RPSList[i].nLTRFrameIdx << endl;
                }
            } else if (ctx->externalRCHints) {
                // Rate Control Feedback
                cout << "Frame: " << frame_num << endl;
                cout << "EncodedBits " << enc_metadata.EncodedFrameBits <<
                    " MinQP " << enc_metadata.FrameMinQP <<
                    " MaxQP " << enc_metadata.FrameMaxQP <<
                    endl;
            } else {
                cout << "Frame " << frame_num <<
                    ": isKeyFrame=" << (int) enc_metadata.KeyFrame <<
                    " AvgQP=" << enc_metadata.AvgQP <<
                    " MinQP=" << enc_metadata.FrameMinQP <<
                    " MaxQP=" << enc_metadata.FrameMaxQP <<
                    " EncodedBits=" << enc_metadata.EncodedFrameBits <<
                    endl;
            }
        }
    }
    if (ctx->dump_mv)
    {
        v4l2_ctrl_videoenc_outputbuf_metadata_MV enc_mv_metadata;
        if (ctx->enc->getMotionVectors(v4l2_buf->index, enc_mv_metadata) == 0)
        {
            uint32_t numMVs = enc_mv_metadata.bufSize / sizeof(MVInfo);
            MVInfo *pInfo = enc_mv_metadata.pMVInfo;

            cout << "Frame " << frame_num << ": Num MVs=" << numMVs << endl;

            for (uint32_t i = 0; i < numMVs; i++, pInfo++)
            {
                cout << i << ": mv_x=" << pInfo->mv_x <<
                    " mv_y=" << pInfo->mv_y <<
                    " weight=" << pInfo->weight <<
                    endl;
            }
        }
    }

    if (enc->capture_plane.qBuffer(*v4l2_buf, NULL) < 0)
    {
        cerr << "Error while Qing buffer at capture plane" << endl;
        abort(ctx);
        return false;
    }

    return true;
}

static int
get_next_parsed_pair(context_t *ctx, char *id, uint32_t *value)
{
    char charval;

    *ctx->runtime_params_str >> *id;
    if (ctx->runtime_params_str->eof())
    {
        return -1;
    }

    charval = ctx->runtime_params_str->peek();
    if (!IS_DIGIT(charval))
    {
        return -1;
    }

    *ctx->runtime_params_str >> *value;

    *ctx->runtime_params_str >> charval;
    if (ctx->runtime_params_str->eof())
    {
        return 0;
    }

    return charval;
}

static int
set_runtime_params(context_t *ctx)
{
    char charval;
    uint32_t intval;
    int ret;

    cout << "Frame " << ctx->next_param_change_frame <<
        ": Changing parameters" << endl;
    while (!ctx->runtime_params_str->eof())
    {
        ret = get_next_parsed_pair(ctx, &charval, &intval);
        TEST_PARSE_ERROR(ret < 0, err);
        switch (charval)
        {
            case 'b':
                ctx->enc->setBitrate(intval);
                cout << "Bitrate = " << intval << endl;
                break;
            case 'r':
            {
                int fps_num = intval;
                TEST_PARSE_ERROR(ret != '/', err);

                ctx->runtime_params_str->seekg(-1, ios::cur);
                ret = get_next_parsed_pair(ctx, &charval, &intval);
                TEST_PARSE_ERROR(ret < 0, err);

                cout << "Framerate = " << fps_num << "/"  << intval << endl;

                ctx->enc->setFrameRate(fps_num, intval);
                break;
            }
            case 'i':
                if (intval > 0)
                {
                    ctx->enc->forceIDR();
                    cout << "Forcing IDR" << endl;
                }
                break;
            default:
                TEST_PARSE_ERROR(true, err);
        }
        switch (ret)
        {
            case 0:
                delete ctx->runtime_params_str;
                ctx->runtime_params_str = NULL;
                return 0;
            case ';':
                return 0;
            case ',':
                break;
            default:
                break;
        }
    }
    return 0;
err:
    cerr << "Skipping further runtime parameter changes" <<endl;
    delete ctx->runtime_params_str;
    ctx->runtime_params_str = NULL;
    return -1;
}

static int
get_next_runtime_param_change_frame(context_t *ctx)
{
    char charval;
    int ret;

    ret = get_next_parsed_pair(ctx, &charval, &ctx->next_param_change_frame);
    if(ret == 0)
    {
        return 0;
    }

    TEST_PARSE_ERROR((ret != ';' && ret != ',') || charval != 'f', err);

    return 0;

err:
    cerr << "Skipping further runtime parameter changes" <<endl;
    delete ctx->runtime_params_str;
    ctx->runtime_params_str = NULL;
    return -1;
}

static void
set_defaults(context_t * ctx)
{
    memset(ctx, 0, sizeof(context_t));

    ctx->bitrate = 4 * 1024 * 1024;
    ctx->profile = V4L2_MPEG_VIDEO_H264_PROFILE_BASELINE;
    ctx->ratecontrol = V4L2_MPEG_VIDEO_BITRATE_MODE_CBR;
    ctx->iframe_interval = 30;
    ctx->externalRPS = false;
    ctx->enableROI = false;
    ctx->bnoIframe = false;
    ctx->bGapsInFrameNumAllowed = false;
    ctx->bReconCrc = false;
    ctx->nH264FrameNumBits = 0;
    ctx->nH265PocLsbBits = 0;
    ctx->idr_interval = 256;
    ctx->level = V4L2_MPEG_VIDEO_H264_LEVEL_5_1;
    ctx->fps_n = 30;
    ctx->fps_d = 1;
    ctx->num_b_frames = (uint32_t) -1;
    ctx->nMinQpI = (uint32_t)QP_RETAIN_VAL;
    ctx->nMaxQpI = (uint32_t)QP_RETAIN_VAL;
    ctx->nMinQpP = (uint32_t)QP_RETAIN_VAL;
    ctx->nMaxQpP = (uint32_t)QP_RETAIN_VAL;
    ctx->nMinQpB = (uint32_t)QP_RETAIN_VAL;
    ctx->nMaxQpB = (uint32_t)QP_RETAIN_VAL;
    ctx->use_gold_crc = false;
    ctx->pBitStreamCrc = NULL;
    ctx->externalRCHints = false;
    ctx->input_metadata = false;
    ctx->sMaxQp = 51;
}

static void
populate_roi_Param(std::ifstream * stream, v4l2_enc_frame_ROI_params *VEnc_ROI_params)
{
    unsigned int ROIIndex = 0;

    if (!stream->eof()) {
        *stream >> VEnc_ROI_params->num_ROI_regions;
        while (ROIIndex < VEnc_ROI_params->num_ROI_regions)
        {
            *stream >> VEnc_ROI_params->ROI_params[ROIIndex].QPdelta;
            *stream >> VEnc_ROI_params->ROI_params[ROIIndex].ROIRect.left;
            *stream >> VEnc_ROI_params->ROI_params[ROIIndex].ROIRect.top;
            *stream >> VEnc_ROI_params->ROI_params[ROIIndex].ROIRect.width;
            *stream >> VEnc_ROI_params->ROI_params[ROIIndex].ROIRect.height;
            ROIIndex++;
        }
    } else {
        cout << "EOF of ROI_param_file & rewind" << endl;
        stream->clear();
        stream->seekg(0);
    }
}

static void
populate_ext_rps_ctrl_Param (std::ifstream * stream, v4l2_enc_frame_ext_rps_ctrl_params *VEnc_ext_rps_ctrl_params)
{
    unsigned int RPSIndex = 0;
    unsigned int temp = 0;

    stream->peek();
restart :
    if (stream->eof()) {
        cout << "EOF of rps_param_file & rewind" << endl;
        stream->clear();
        stream->seekg(0);
    }
    if (!stream->eof()) {
        *stream >> VEnc_ext_rps_ctrl_params->nFrameId;
        if (stream->eof())
            goto restart;
        *stream >> temp;
        VEnc_ext_rps_ctrl_params->bRefFrame = ((temp)?true:false);
        *stream >> temp;
        VEnc_ext_rps_ctrl_params->bLTRefFrame = ((temp)?true:false);
        *stream >> VEnc_ext_rps_ctrl_params->nMaxRefFrames;
        *stream >> VEnc_ext_rps_ctrl_params->nActiveRefFrames;
        *stream >> VEnc_ext_rps_ctrl_params->nCurrentRefFrameId;
        while (RPSIndex < VEnc_ext_rps_ctrl_params->nActiveRefFrames)
        {
            *stream >> VEnc_ext_rps_ctrl_params->RPSList[RPSIndex].nFrameId;
            *stream >> temp;
            VEnc_ext_rps_ctrl_params->RPSList[RPSIndex].bLTRefFrame = ((temp)?true:false);
            RPSIndex++;
        }
    }
}

static void
populate_ext_rate_ctrl_Param(std::ifstream * stream, v4l2_enc_frame_ext_rate_ctrl_params *VEnc_ext_rate_ctrl_params)
{
    stream->peek();
restart:
    if (stream->eof()) {
        cout << "EOF of hints_param_file & rewind" << endl;
        stream->clear();
        stream->seekg(0);
    }
    if (!stream->eof()) {
        *stream >> VEnc_ext_rate_ctrl_params->nTargetFrameBits;
        if (stream->eof())
            goto restart;
        *stream >> VEnc_ext_rate_ctrl_params->nFrameQP;
        *stream >> VEnc_ext_rate_ctrl_params->nFrameMinQp;
        *stream >> VEnc_ext_rate_ctrl_params->nFrameMaxQp;
        *stream >> VEnc_ext_rate_ctrl_params->nMaxQPDeviation;
    }
}

int
main(int argc, char *argv[])
{
    context_t ctx;
    int ret = 0;
    int error = 0;
    bool eos = false;

    set_defaults(&ctx);

    ret = parse_csv_args(&ctx, argc, argv);
    TEST_ERROR(ret < 0, "Error parsing commandline arguments", cleanup);

    if (ctx.runtime_params_str)
    {
        get_next_runtime_param_change_frame(&ctx);
    }

    if (ctx.encoder_pixfmt == V4L2_PIX_FMT_H265)
    {
        TEST_ERROR(ctx.width < 144 || ctx.height < 144, "Height/Width should be"
            " > 144 for H.265", cleanup);
    }

    if (ctx.use_gold_crc)
    {
        ctx.pBitStreamCrc = InitCrc(CRC32_POLYNOMIAL);
        TEST_ERROR(!ctx.pBitStreamCrc, "InitCrc failed", cleanup);
    }

    ctx.in_file = new ifstream(ctx.in_file_path);
    TEST_ERROR(!ctx.in_file->is_open(), "Could not open input file", cleanup);

    ctx.out_file = new ofstream(ctx.out_file_path);
    TEST_ERROR(!ctx.out_file->is_open(), "Could not open output file", cleanup);

    if (ctx.ROI_Param_file_path) {
        ctx.roi_Param_file = new ifstream(ctx.ROI_Param_file_path);
        TEST_ERROR(!ctx.roi_Param_file->is_open(), "Could not open roi param file", cleanup);
    }

    if (ctx.Recon_Ref_file_path) {
        ctx.recon_Ref_file = new ifstream(ctx.Recon_Ref_file_path);
        TEST_ERROR(!ctx.recon_Ref_file->is_open(), "Could not open recon crc reference file", cleanup);
    }

    if (ctx.RPS_Param_file_path) {
        ctx.rps_Param_file = new ifstream(ctx.RPS_Param_file_path);
        TEST_ERROR(!ctx.rps_Param_file->is_open(), "Could not open rps param file", cleanup);
    }

    if (ctx.hints_Param_file_path) {
        ctx.hints_Param_file = new ifstream(ctx.hints_Param_file_path);
        TEST_ERROR(!ctx.hints_Param_file->is_open(), "Could not open hints param file", cleanup);
    }

    ctx.enc = NvVideoEncoder::createVideoEncoder("enc0");
    TEST_ERROR(!ctx.enc, "Could not create encoder", cleanup);

    // It is necessary that Capture Plane format be set before Output Plane
    // format.
    // Set encoder capture plane format. It is necessary to set width and
    // height on thr capture plane as well
    ret =
        ctx.enc->setCapturePlaneFormat(ctx.encoder_pixfmt, ctx.width,
                                      ctx.height, 2 * 1024 * 1024);
    TEST_ERROR(ret < 0, "Could not set output plane format", cleanup);

    // Set encoder output plane format
    ret =
        ctx.enc->setOutputPlaneFormat(V4L2_PIX_FMT_YUV420M, ctx.width,
                                      ctx.height);
    TEST_ERROR(ret < 0, "Could not set output plane format", cleanup);

    ret = ctx.enc->setBitrate(ctx.bitrate);
    TEST_ERROR(ret < 0, "Could not set encoder bitrate", cleanup);

    ret = ctx.enc->setProfile(ctx.profile);
    TEST_ERROR(ret < 0, "Could not set encoder profile", cleanup);

    if (ctx.encoder_pixfmt == V4L2_PIX_FMT_H264)
    {
        ret = ctx.enc->setLevel(ctx.level);
        TEST_ERROR(ret < 0, "Could not set encoder level", cleanup);
    }

    ret = ctx.enc->setRateControlMode(ctx.ratecontrol);
    TEST_ERROR(ret < 0, "Could not set encoder rate control mode", cleanup);

    ret = ctx.enc->setIDRInterval(ctx.idr_interval);
    TEST_ERROR(ret < 0, "Could not set encoder IDR interval", cleanup);

    ret = ctx.enc->setIFrameInterval(ctx.iframe_interval);
    TEST_ERROR(ret < 0, "Could not set encoder I-Frame interval", cleanup);

    ret = ctx.enc->setFrameRate(ctx.fps_n, ctx.fps_d);
    TEST_ERROR(ret < 0, "Could not set framerate", cleanup);

    if (ctx.temporal_tradeoff_level)
    {
        ret = ctx.enc->setTemporalTradeoff(ctx.temporal_tradeoff_level);
        TEST_ERROR(ret < 0, "Could not set temporal tradeoff level", cleanup);
    }

    if (ctx.slice_length)
    {
        ret = ctx.enc->setSliceLength(ctx.slice_length_type,
                ctx.slice_length);
        TEST_ERROR(ret < 0, "Could not set slice length params", cleanup);
    }

    if (ctx.virtual_buffer_size)
    {
        ret = ctx.enc->setVirtualBufferSize(ctx.virtual_buffer_size);
        TEST_ERROR(ret < 0, "Could not set virtual buffer size", cleanup);
    }

    if (ctx.num_reference_frames)
    {
        ret = ctx.enc->setNumReferenceFrames(ctx.num_reference_frames);
        TEST_ERROR(ret < 0, "Could not set num reference frames", cleanup);
    }

    if (ctx.slice_intrarefresh_interval)
    {
        ret = ctx.enc->setSliceIntrarefresh(ctx.slice_intrarefresh_interval);
        TEST_ERROR(ret < 0, "Could not set slice intrarefresh interval", cleanup);
    }

    if (ctx.insert_sps_pps_at_idr)
    {
        ret = ctx.enc->setInsertSpsPpsAtIdrEnabled(true);
        TEST_ERROR(ret < 0, "Could not set insertSPSPPSAtIDR", cleanup);
    }

    if (ctx.num_b_frames != (uint32_t) -1)
    {
        ret = ctx.enc->setNumBFrames(ctx.num_b_frames);
        TEST_ERROR(ret < 0, "Could not set number of B Frames", cleanup);
    }

    if ((ctx.nMinQpI != (uint32_t)QP_RETAIN_VAL) ||
        (ctx.nMaxQpI != (uint32_t)QP_RETAIN_VAL) ||
        (ctx.nMinQpP != (uint32_t)QP_RETAIN_VAL) ||
        (ctx.nMaxQpP != (uint32_t)QP_RETAIN_VAL) ||
        (ctx.nMinQpB != (uint32_t)QP_RETAIN_VAL) ||
        (ctx.nMaxQpB != (uint32_t)QP_RETAIN_VAL))
    {
        ret = ctx.enc->setQpRange(ctx.nMinQpI, ctx.nMaxQpI, ctx.nMinQpP,
                ctx.nMaxQpP, ctx.nMinQpB, ctx.nMaxQpB);
        TEST_ERROR(ret < 0, "Could not set quantization parameters", cleanup);
    }

    if (ctx.dump_mv)
    {
        ret = ctx.enc->enableMotionVectorReporting();
        TEST_ERROR(ret < 0, "Could not enable motion vector reporting", cleanup);
    }

    if (ctx.bnoIframe) {
        ctx.iframe_interval = ((1<<31) + 1); /* TODO: how can we do this properly */
        ret = ctx.enc->setIFrameInterval(ctx.iframe_interval);
        TEST_ERROR(ret < 0, "Could not set encoder I-Frame interval", cleanup);
    }

    if (ctx.enableROI) {
        v4l2_enc_enable_roi_param VEnc_enable_ext_roi_ctrl;

        VEnc_enable_ext_roi_ctrl.bEnableROI = ctx.enableROI;
        ret = ctx.enc->enableROI(VEnc_enable_ext_roi_ctrl);
        TEST_ERROR(ret < 0, "Could not enable ROI", cleanup);
    }

    if (ctx.bReconCrc) {
        v4l2_enc_enable_reconcrc_param VEnc_enable_recon_crc_ctrl;

        VEnc_enable_recon_crc_ctrl.bEnableReconCRC = ctx.bReconCrc;
        ret = ctx.enc->enableReconCRC(VEnc_enable_recon_crc_ctrl);
        TEST_ERROR(ret < 0, "Could not enable Recon CRC", cleanup);
    }

    if (ctx.externalRPS) {
        v4l2_enc_enable_ext_rps_ctr VEnc_enable_ext_rps_ctrl;

        VEnc_enable_ext_rps_ctrl.bEnableExternalRPS = ctx.externalRPS;
        if (ctx.encoder_pixfmt == V4L2_PIX_FMT_H264) {
            VEnc_enable_ext_rps_ctrl.bGapsInFrameNumAllowed = ctx.bGapsInFrameNumAllowed;
            VEnc_enable_ext_rps_ctrl.nH264FrameNumBits = ctx.nH264FrameNumBits;
        }
        if (ctx.encoder_pixfmt == V4L2_PIX_FMT_H265) {
            VEnc_enable_ext_rps_ctrl.nH265PocLsbBits = ctx.nH265PocLsbBits;
        }
        ret = ctx.enc->enableExternalRPS(VEnc_enable_ext_rps_ctrl);
        TEST_ERROR(ret < 0, "Could not enable external RPS", cleanup);
    }

    if (ctx.externalRCHints) {
        v4l2_enc_enable_ext_rate_ctr VEnc_enable_ext_rate_ctrl;

        VEnc_enable_ext_rate_ctrl.bEnableExternalPictureRC = ctx.externalRCHints;
        VEnc_enable_ext_rate_ctrl.nsessionMaxQP = ctx.sMaxQp;

        ret = ctx.enc->enableExternalRC(VEnc_enable_ext_rate_ctrl);
        TEST_ERROR(ret < 0, "Could not enable external RC", cleanup);
    }

    // Query, Export and Map the output plane buffers so that we can read
    // raw data into the buffers
    ret = ctx.enc->output_plane.setupPlane(V4L2_MEMORY_MMAP, 10, true, false);
    TEST_ERROR(ret < 0, "Could not setup output plane", cleanup);

    // Query, Export and Map the output plane buffers so that we can write
    // encoded data from the buffers
    ret = ctx.enc->capture_plane.setupPlane(V4L2_MEMORY_MMAP, 10, true, false);
    TEST_ERROR(ret < 0, "Could not setup capture plane", cleanup);

    // output plane STREAMON
    ret = ctx.enc->output_plane.setStreamStatus(true);
    TEST_ERROR(ret < 0, "Error in output plane streamon", cleanup);

    // capture plane STREAMON
    ret = ctx.enc->capture_plane.setStreamStatus(true);
    TEST_ERROR(ret < 0, "Error in capture plane streamon", cleanup);

    ctx.enc->capture_plane.
        setDQThreadCallback(encoder_capture_plane_dq_callback);

    // startDQThread starts a thread internally which calls the
    // encoder_capture_plane_dq_callback whenever a buffer is dequeued
    // on the plane
    ctx.enc->capture_plane.startDQThread(&ctx);

    // Enqueue all the empty capture plane buffers
    for (uint32_t i = 0; i < ctx.enc->capture_plane.getNumBuffers(); i++)
    {
        struct v4l2_buffer v4l2_buf;
        struct v4l2_plane planes[MAX_PLANES];

        memset(&v4l2_buf, 0, sizeof(v4l2_buf));
        memset(planes, 0, MAX_PLANES * sizeof(struct v4l2_plane));

        v4l2_buf.index = i;
        v4l2_buf.m.planes = planes;

        ret = ctx.enc->capture_plane.qBuffer(v4l2_buf, NULL);
        if (ret < 0)
        {
            cerr << "Error while queueing buffer at capture plane" << endl;
            abort(&ctx);
            goto cleanup;
        }
    }

    // Read video frame and queue all the output plane buffers
    for (uint32_t i = 0; i < ctx.enc->output_plane.getNumBuffers(); i++)
    {
        struct v4l2_buffer v4l2_buf;
        struct v4l2_plane planes[MAX_PLANES];
        NvBuffer *buffer = ctx.enc->output_plane.getNthBuffer(i);

        memset(&v4l2_buf, 0, sizeof(v4l2_buf));
        memset(planes, 0, MAX_PLANES * sizeof(struct v4l2_plane));

        v4l2_buf.index = i;
        v4l2_buf.m.planes = planes;

        if (read_video_frame(ctx.in_file, *buffer) < 0)
        {
            cerr << "Could not read complete frame from input file" << endl;
            v4l2_buf.m.planes[0].bytesused = 0;
        }

        if (ctx.runtime_params_str &&
            (ctx.enc->output_plane.getTotalQueuedBuffers() ==
                ctx.next_param_change_frame))
        {
            set_runtime_params(&ctx);
            if (ctx.runtime_params_str)
                get_next_runtime_param_change_frame(&ctx);
        }

        if (ctx.input_metadata)
        {
            v4l2_ctrl_videoenc_input_metadata VEnc_imeta_param;
            v4l2_enc_frame_ROI_params VEnc_ROI_params;
            v4l2_enc_frame_ReconCRC_params VEnc_ReconCRC_params;
            v4l2_enc_frame_ext_rps_ctrl_params VEnc_ext_rps_ctrl_params;
            v4l2_enc_frame_ext_rate_ctrl_params VEnc_ext_rate_ctrl_params;
            VEnc_imeta_param.flag = 0;

            if (ctx.ROI_Param_file_path)
            {
                if (ctx.enableROI) {
                    VEnc_imeta_param.flag |= V4L2_ENC_INPUT_ROI_PARAM_FLAG;
                    VEnc_imeta_param.VideoEncROIParams = &VEnc_ROI_params;

                    populate_roi_Param(ctx.roi_Param_file, VEnc_imeta_param.VideoEncROIParams);
                }
            }

            if (ctx.bReconCrc)
            {
                VEnc_imeta_param.flag |= V4L2_ENC_INPUT_RECONCRC_PARAM_FLAG;

                VEnc_ReconCRC_params.ReconCRCRect.left = ctx.rl;
                VEnc_ReconCRC_params.ReconCRCRect.top = ctx.rt;
                VEnc_ReconCRC_params.ReconCRCRect.width = ctx.rw;
                VEnc_ReconCRC_params.ReconCRCRect.height = ctx.rh;

                VEnc_imeta_param.VideoReconCRCParams = &VEnc_ReconCRC_params;
            }

            if (ctx.RPS_Param_file_path)
            {
                if (ctx.externalRPS) {
                    VEnc_imeta_param.flag |= V4L2_ENC_INPUT_RPS_PARAM_FLAG;
                    VEnc_imeta_param.VideoEncRPSParams = &VEnc_ext_rps_ctrl_params;

                    populate_ext_rps_ctrl_Param(ctx.rps_Param_file, VEnc_imeta_param.VideoEncRPSParams);
                }
            }

            if (ctx.hints_Param_file_path)
            {
                if (ctx.externalRCHints) {
                    VEnc_imeta_param.flag |= V4L2_ENC_INPUT_RC_PARAM_FLAG;
                    VEnc_imeta_param.VideoEncExtRCParams = &VEnc_ext_rate_ctrl_params;

                    populate_ext_rate_ctrl_Param(ctx.hints_Param_file, VEnc_imeta_param.VideoEncExtRCParams);
                }
            }

            if (VEnc_imeta_param.flag)
            {
                ctx.enc->SetInputMetaParams(v4l2_buf.index, VEnc_imeta_param);
                v4l2_buf.reserved2 = v4l2_buf.index;
            }
        }

        ret = ctx.enc->output_plane.qBuffer(v4l2_buf, NULL);
        if (ret < 0)
        {
            cerr << "Error while queueing buffer at output plane" << endl;
            abort(&ctx);
            goto cleanup;
        }

        if (v4l2_buf.m.planes[0].bytesused == 0)
        {
            cerr << "File read complete." << endl;
            eos = true;
            break;
        }
    }

    // Keep reading input till EOS is reached
    while (!ctx.got_error && !ctx.enc->isInError() && !eos)
    {
        struct v4l2_buffer v4l2_buf;
        struct v4l2_plane planes[MAX_PLANES];
        NvBuffer *buffer;

        memset(&v4l2_buf, 0, sizeof(v4l2_buf));
        memset(planes, 0, sizeof(planes));

        v4l2_buf.m.planes = planes;

        if (ctx.enc->output_plane.dqBuffer(v4l2_buf, &buffer, NULL, 10) < 0)
        {
            cerr << "ERROR while DQing buffer at output plane" << endl;
            abort(&ctx);
            goto cleanup;
        }

        if (ctx.runtime_params_str &&
            (ctx.enc->output_plane.getTotalQueuedBuffers() ==
                ctx.next_param_change_frame))
        {
            set_runtime_params(&ctx);
            if (ctx.runtime_params_str)
                get_next_runtime_param_change_frame(&ctx);
        }
        if (read_video_frame(ctx.in_file, *buffer) < 0)
        {
            cerr << "Could not read complete frame from input file" << endl;
            v4l2_buf.m.planes[0].bytesused = 0;
        }

        if (ctx.input_metadata)
        {
            v4l2_ctrl_videoenc_input_metadata VEnc_imeta_param;
            v4l2_enc_frame_ROI_params VEnc_ROI_params;
            v4l2_enc_frame_ReconCRC_params VEnc_ReconCRC_params;
            v4l2_enc_frame_ext_rps_ctrl_params VEnc_ext_rps_ctrl_params;
            v4l2_enc_frame_ext_rate_ctrl_params VEnc_ext_rate_ctrl_params;
            VEnc_imeta_param.flag = 0;

            if (ctx.ROI_Param_file_path)
            {
                if (ctx.enableROI) {
                    VEnc_imeta_param.flag |= V4L2_ENC_INPUT_ROI_PARAM_FLAG;
                    VEnc_imeta_param.VideoEncROIParams = &VEnc_ROI_params;

                    populate_roi_Param(ctx.roi_Param_file, VEnc_imeta_param.VideoEncROIParams);
                }
            }

            if (ctx.bReconCrc)
            {
                VEnc_imeta_param.flag |= V4L2_ENC_INPUT_RECONCRC_PARAM_FLAG;

                VEnc_ReconCRC_params.ReconCRCRect.left = ctx.rl;
                VEnc_ReconCRC_params.ReconCRCRect.top = ctx.rt;
                VEnc_ReconCRC_params.ReconCRCRect.width = ctx.rw;
                VEnc_ReconCRC_params.ReconCRCRect.height = ctx.rh;

                VEnc_imeta_param.VideoReconCRCParams = &VEnc_ReconCRC_params;
            }

            if (ctx.RPS_Param_file_path)
            {
                if (ctx.externalRPS) {
                    VEnc_imeta_param.flag |= V4L2_ENC_INPUT_RPS_PARAM_FLAG;
                    VEnc_imeta_param.VideoEncRPSParams = &VEnc_ext_rps_ctrl_params;

                    populate_ext_rps_ctrl_Param(ctx.rps_Param_file, VEnc_imeta_param.VideoEncRPSParams);
                }
            }

            if (ctx.hints_Param_file_path)
            {
                if (ctx.externalRCHints) {
                    VEnc_imeta_param.flag |= V4L2_ENC_INPUT_RC_PARAM_FLAG;
                    VEnc_imeta_param.VideoEncExtRCParams = &VEnc_ext_rate_ctrl_params;

                    populate_ext_rate_ctrl_Param(ctx.hints_Param_file, VEnc_imeta_param.VideoEncExtRCParams);

                }
            }

            if (VEnc_imeta_param.flag)
            {
                ctx.enc->SetInputMetaParams(v4l2_buf.index, VEnc_imeta_param);
                v4l2_buf.reserved2 = v4l2_buf.index;
            }
        }

        ret = ctx.enc->output_plane.qBuffer(v4l2_buf, NULL);
        if (ret < 0)
        {
            cerr << "Error while queueing buffer at output plane" << endl;
            abort(&ctx);
            goto cleanup;
        }

        if (v4l2_buf.m.planes[0].bytesused == 0)
        {
            cerr << "File read complete." << endl;
            eos = true;
            break;
        }
    }

    // Wait till capture plane DQ Thread finishes
    // i.e. all the capture plane buffers are dequeued
    ctx.enc->capture_plane.waitForDQThread(-1);

cleanup:
    if (ctx.enc && ctx.enc->isInError())
    {
        cerr << "Encoder is in error" << endl;
        error = 1;
    }
    if (ctx.got_error)
    {
        error = 1;
    }

    if (ctx.pBitStreamCrc)
    {
        char *pgold_crc = ctx.gold_crc;
        Crc *pout_crc= ctx.pBitStreamCrc;
        char StrCrcValue[20];
        snprintf (StrCrcValue, 20, "%u", pout_crc->CrcValue);
        // Remove CRLF from end of CRC, if present
        do {
               unsigned int len = strlen(pgold_crc);
               if (len == 0) break;
               if (pgold_crc[len-1] == '\n')
                   pgold_crc[len-1] = '\0';
               else if (pgold_crc[len-1] == '\r')
                   pgold_crc[len-1] = '\0';
               else
                   break;
        } while(1);

        if (strcmp (StrCrcValue, pgold_crc))
        {
            cout << "======================" << endl;
            cout << "video_encode: CRC FAILED" << endl;
            cout << "======================" << endl;
            cout << "Encoded CRC: " << StrCrcValue << " Gold CRC: " << pgold_crc << endl;
            error = 1;
        }
        else
        {
            cout << "======================" << endl;
            cout << "video_encode: CRC PASSED" << endl;
            cout << "======================" << endl;
        }

        CloseCrc(&ctx.pBitStreamCrc);
    }

    delete ctx.enc;
    delete ctx.in_file;
    delete ctx.out_file;
    delete ctx.roi_Param_file;
    delete ctx.recon_Ref_file;
    delete ctx.rps_Param_file;
    delete ctx.hints_Param_file;

    free(ctx.in_file_path);
    free(ctx.out_file_path);
    free(ctx.ROI_Param_file_path);
    free(ctx.Recon_Ref_file_path);
    free(ctx.RPS_Param_file_path);
    free(ctx.hints_Param_file_path);
    delete ctx.runtime_params_str;

    if (error)
    {
        cout << "App run failed" << endl;
    }
    else
    {
        cout << "App run was successful" << endl;
    }
    return -error;
}
