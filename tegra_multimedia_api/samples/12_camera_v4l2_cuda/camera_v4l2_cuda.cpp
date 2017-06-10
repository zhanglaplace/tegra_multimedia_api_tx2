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

#include <stdio.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <errno.h>
#include <stdlib.h>
#include <signal.h>
#include <poll.h>

#include "NvVideoConverter.h"
#include "NvEglRenderer.h"
#include "NvUtils.h"
#include "NvCudaProc.h"
#include "nvbuf_utils.h"

#include "camera_v4l2_cuda.h"

static bool quit = false;

using namespace std;

static void
print_usage(void)
{
    printf("\n\tUsage: camera_v4l2_cuda [OPTIONS]\n\n"
           "\tExample: \n"
           "\t./camera_v4l2_cuda -d /dev/video0 -s 640x480 -f YUYV -n 30 -c\n\n"
           "\tSupported options:\n"
           "\t-d\t\tSet V4l2 video device node\n"
           "\t-s\t\tSet output resolution of video device\n"
           "\t-f\t\tSet output pixel format of video device (supports only YUYV/YVYU/UYVY/VYUY)\n"
           "\t-r\t\tSet renderer frame rate (30 fps by default)\n"
           "\t-n\t\tSave the n-th frame before VIC processing\n"
           "\t-c\t\tEnable CUDA aglorithm (draw a black box in the upper left corner)\n"
           "\t-v\t\tEnable verbose message\n"
           "\t-h\t\tPrint this usage\n\n"
           "\tNOTE: It runs infinitely until you terminate it with <ctrl+c>\n");
}

static bool
parse_cmdline(context_t * ctx, int argc, char **argv)
{
    int c;

    if (argc < 2)
    {
        print_usage();
        exit(EXIT_SUCCESS);
    }

    while ((c = getopt(argc, argv, "d:s:f:r:n:cvh")) != -1)
    {
        switch (c)
        {
            case 'd':
                ctx->cam_devname = optarg;
                break;
            case 's':
                if (sscanf(optarg, "%dx%d",
                            &ctx->cam_w, &ctx->cam_h) != 2)
                {
                    print_usage();
                    return false;
                }
                break;
            case 'f':
                if (strcmp(optarg, "YUYV") == 0)
                    ctx->cam_pixfmt = V4L2_PIX_FMT_YUYV;
                else if (strcmp(optarg, "YVYU") == 0)
                    ctx->cam_pixfmt = V4L2_PIX_FMT_YVYU;
                else if (strcmp(optarg, "VYUY") == 0)
                    ctx->cam_pixfmt = V4L2_PIX_FMT_VYUY;
                else if (strcmp(optarg, "UYVY") == 0)
                    ctx->cam_pixfmt = V4L2_PIX_FMT_UYVY;
                else
                {
                    print_usage();
                    return false;
                }
                sprintf(ctx->cam_file, "camera.%s", optarg);
                break;
            case 'r':
                ctx->fps = strtol(optarg, NULL, 10);
                break;
            case 'n':
                ctx->save_n_frame = strtol(optarg, NULL, 10);
                break;
            case 'c':
                ctx->enable_cuda = true;
                break;
            case 'v':
                ctx->enable_verbose = true;
                break;
            case 'h':
                print_usage();
                exit(EXIT_SUCCESS);
                break;
            default:
                print_usage();
                return false;
        }
    }

    return true;
}

static void
set_defaults(context_t * ctx)
{
    memset(ctx, 0, sizeof(context_t));

    ctx->cam_devname = "/dev/video0";
    ctx->cam_fd = -1;
    ctx->cam_pixfmt = V4L2_PIX_FMT_YUYV;
    ctx->cam_w = 640;
    ctx->cam_h = 480;
    ctx->frame = 0;
    ctx->save_n_frame = 0;

    ctx->conv = NULL;
    ctx->vic_pixfmt = V4L2_PIX_FMT_YUV420M;
    ctx->vic_flip = (enum v4l2_flip_method) -1;
    ctx->vic_interpolation = (enum v4l2_interpolation_method) -1;
    ctx->vic_tnr = (enum v4l2_tnr_algorithm) -1;

    ctx->g_buff = NULL;
    ctx->renderer = NULL;
    ctx->got_error = false;
    ctx->fps = 30;

    ctx->conv_output_plane_buf_queue = new queue < nv_buffer * >;
    pthread_mutex_init(&ctx->queue_lock, NULL);
    pthread_cond_init(&ctx->queue_cond, NULL);

    ctx->enable_cuda = false;
    ctx->egl_image = NULL;
    ctx->egl_display = EGL_NO_DISPLAY;

    ctx->enable_verbose = false;
}

static nv_color_fmt nvcolor_fmt[] =
{
    // TODO add more pixel format mapping
    {V4L2_PIX_FMT_UYVY, NvBufferColorFormat_UYVY},
    {V4L2_PIX_FMT_VYUY, NvBufferColorFormat_VYUY},
    {V4L2_PIX_FMT_YUYV, NvBufferColorFormat_YUYV},
    {V4L2_PIX_FMT_YVYU, NvBufferColorFormat_YVYU},
};

static NvBufferColorFormat
get_nvbuff_color_fmt(unsigned int v4l2_pixfmt)
{
  for (unsigned i = 0; i < sizeof(nvcolor_fmt); i++)
  {
    if (v4l2_pixfmt == nvcolor_fmt[i].v4l2_pixfmt)
      return nvcolor_fmt[i].nvbuff_color;
  }

  return NvBufferColorFormat_Invalid;
}

static bool
save_frame_to_file(context_t * ctx, struct v4l2_buffer * buf)
{
    int file;

    file = open(ctx->cam_file, O_CREAT | O_WRONLY | O_APPEND | O_TRUNC,
            S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP | S_IROTH | S_IWOTH);

    if (-1 == file)
        ERROR_RETURN("Failed to open file for frame saving");

    if (-1 == write(file, ctx->g_buff[buf->index].start,
                ctx->g_buff[buf->index].size))
        ERROR_RETURN("Failed to write frame into file");

    close(file);

    return true;
}

static bool
camera_initialize(context_t * ctx)
{
    struct v4l2_format fmt;

    // Open camera device
    ctx->cam_fd = open(ctx->cam_devname, O_RDWR);
    if (ctx->cam_fd == -1)
        ERROR_RETURN("Failed to open camera device %s: %s (%d)",
                ctx->cam_devname, strerror(errno), errno);

    // Set camera output format
    memset(&fmt, 0, sizeof(fmt));
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width = ctx->cam_w;
    fmt.fmt.pix.height = ctx->cam_h;
    fmt.fmt.pix.pixelformat = ctx->cam_pixfmt;
    fmt.fmt.pix.field = V4L2_FIELD_INTERLACED;
    if (ioctl(ctx->cam_fd, VIDIOC_S_FMT, &fmt) < 0)
        ERROR_RETURN("Failed to set camera output format: %s (%d)",
                strerror(errno), errno);

    // Get the real format in case the desired is not supported
    memset(&fmt, 0, sizeof fmt);
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (ioctl(ctx->cam_fd, VIDIOC_G_FMT, &fmt) < 0)
        ERROR_RETURN("Failed to get camera output format: %s (%d)",
                strerror(errno), errno);
    if (fmt.fmt.pix.width != ctx->cam_w ||
            fmt.fmt.pix.height != ctx->cam_h ||
            fmt.fmt.pix.pixelformat != ctx->cam_pixfmt)
    {
        WARN("The desired format is not supported");
        ctx->cam_w = fmt.fmt.pix.width;
        ctx->cam_h = fmt.fmt.pix.height;
        ctx->cam_pixfmt =fmt.fmt.pix.pixelformat;
    }

    INFO("Camera ouput format: (%d x %d)  stride: %d, imagesize: %d",
            fmt.fmt.pix.width,
            fmt.fmt.pix.height,
            fmt.fmt.pix.bytesperline,
            fmt.fmt.pix.sizeimage);

    return true;
}

static bool
vic_initialize(context_t * ctx)
{
    // Create VIC (VIdeo Converter) instance
    ctx->conv = NvVideoConverter::createVideoConverter("conv");
    if (ctx->conv == NULL)
        ERROR_RETURN("Failed to create video converter");

    if (ctx->vic_flip != -1 && ctx->conv->setFlipMethod(ctx->vic_flip) < 0)
        ERROR_RETURN("Failed to set flip method");

    if (ctx->vic_interpolation != -1 &&
            ctx->conv->setInterpolationMethod(ctx->vic_interpolation) < 0)
            ERROR_RETURN("Failed to set interpolation method");

    if (ctx->vic_tnr != -1 && ctx->conv->setTnrAlgorithm(ctx->vic_tnr) < 0)
            ERROR_RETURN("Failed to set tnr algorithm");

    // Set up VIC output plane format
    if (ctx->conv->setOutputPlaneFormat(ctx->cam_pixfmt, ctx->cam_w,
                ctx->cam_h, V4L2_NV_BUFFER_LAYOUT_PITCH) < 0)
        ERROR_RETURN("Failed to set up VIC output plane format");

    // Set up VIC capture plane format
    // The target format can be reconfigured from set_defaults()
    if (ctx->conv->setCapturePlaneFormat(ctx->vic_pixfmt, ctx->cam_w,
                ctx->cam_h, V4L2_NV_BUFFER_LAYOUT_PITCH) < 0)
        ERROR_RETURN("Failed to set up VIC capture plane format");

    // Allocate VIC output plane
    if (ctx->conv->output_plane.setupPlane(V4L2_MEMORY_DMABUF,
                V4L2_BUFFERS_NUM, false, false) < 0)
        ERROR_RETURN("Failed to allocate VIC output plane");

    // Allocate VIC capture plane
    if (ctx->conv->capture_plane.setupPlane(V4L2_MEMORY_MMAP,
                V4L2_BUFFERS_NUM, true, false) < 0)
        ERROR_RETURN("Failed to allocate VIC capture plane");

    return true;
}

static bool
display_initialize(context_t * ctx)
{
    // Create EGL renderer
    ctx->renderer = NvEglRenderer::createEglRenderer("renderer0",
            ctx->cam_w, ctx->cam_h, 0, 0);
    if (!ctx->renderer)
        ERROR_RETURN("Failed to create EGL renderer");
    ctx->renderer->setFPS(ctx->fps);

    if (ctx->enable_cuda)
    {
        // Get defalut EGL display
        ctx->egl_display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
        if (ctx->egl_display == EGL_NO_DISPLAY)
            ERROR_RETURN("Failed to get EGL display connection");

        // Init EGL display connection
        if (!eglInitialize(ctx->egl_display, NULL, NULL))
            ERROR_RETURN("Failed to initialize EGL display connection");
    }

    return true;
}

static bool
init_components(context_t * ctx)
{
    if (!camera_initialize(ctx))
        ERROR_RETURN("Failed to initialize camera device");

    if (!vic_initialize(ctx))
        ERROR_RETURN("Failed to initialize video converter");

    if (!display_initialize(ctx))
        ERROR_RETURN("Failed to initialize display");

    INFO("Initialize v4l2 components successfully");
    return true;
}

static bool
request_camera_buff(context_t *ctx)
{
    // Request camera v4l2 buffer
    struct v4l2_requestbuffers rb;
    memset(&rb, 0, sizeof(rb));
    rb.count = V4L2_BUFFERS_NUM;
    rb.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    rb.memory = V4L2_MEMORY_DMABUF;
    if (ioctl(ctx->cam_fd, VIDIOC_REQBUFS, &rb) < 0)
        ERROR_RETURN("Failed to request v4l2 buffers: %s (%d)",
                strerror(errno), errno);
    if (rb.count != V4L2_BUFFERS_NUM)
        ERROR_RETURN("V4l2 buffer number is not as desired");

    for (unsigned int index = 0; index < V4L2_BUFFERS_NUM; index++)
    {
        struct v4l2_buffer buf;

        // Query camera v4l2 buf length
        memset(&buf, 0, sizeof buf);
        buf.index = index;
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_DMABUF;

        if (ioctl(ctx->cam_fd, VIDIOC_QUERYBUF, &buf) < 0)
            ERROR_RETURN("Failed to query buff: %s (%d)",
                    strerror(errno), errno);

        // TODO add support for multi-planer
        // Enqueue empty v4l2 buff into camera capture plane
        buf.m.fd = (unsigned long)ctx->g_buff[index].dmabuff_fd;
        if (buf.length != ctx->g_buff[index].size)
        {
            WARN("Camera v4l2 buf length is not expected");
            ctx->g_buff[index].size = buf.length;
        }

        if (ioctl(ctx->cam_fd, VIDIOC_QBUF, &buf) < 0)
            ERROR_RETURN("Failed to enqueue buffers: %s (%d)",
                    strerror(errno), errno);
    }

    return true;
}

static bool
enqueue_vic_buff(context_t *ctx)
{
    // Enqueue empty buffer into VIC output plane
    for (unsigned int index = 0;
            index < ctx->conv->output_plane.getNumBuffers(); index++)
    {
        struct v4l2_buffer v4l2_buf;
        struct v4l2_plane planes[MAX_PLANES];

        memset(&v4l2_buf, 0, sizeof(v4l2_buf));
        memset(planes, 0, MAX_PLANES * sizeof(struct v4l2_plane));

        v4l2_buf.index = index;
        v4l2_buf.m.planes = planes;

        if (ctx->conv->output_plane.qBuffer(v4l2_buf, NULL) < 0)
            ERROR_RETURN("Failed to enqueue empty buffer into VIC output plane");
    }

    // Enqueue empty buffer into VIC capture plane
    for (unsigned int index = 0;
            index < ctx->conv->capture_plane.getNumBuffers(); index++)
    {
        struct v4l2_buffer v4l2_buf;
        struct v4l2_plane planes[MAX_PLANES];

        memset(&v4l2_buf, 0, sizeof(v4l2_buf));
        memset(planes, 0, MAX_PLANES * sizeof(struct v4l2_plane));

        v4l2_buf.index = index;
        v4l2_buf.m.planes = planes;

        if (ctx->conv->capture_plane.qBuffer(v4l2_buf, NULL) < 0)
            ERROR_RETURN("Failed to enqueue empty buffer into VIC capture plane");
    }

    return true;
}

static bool
prepare_buffers(context_t * ctx)
{
    // Allocate global buffer context
    ctx->g_buff = (nv_buffer *)malloc(V4L2_BUFFERS_NUM * sizeof(nv_buffer));
    if (ctx->g_buff == NULL)
        ERROR_RETURN("Failed to allocate global buffer context");

    // Create buffer and share it with camera and VIC output plane
    for (unsigned int index = 0; index < V4L2_BUFFERS_NUM; index++)
    {
        int fd;
        NvBufferParams params = {0};

        if (-1 == NvBufferCreate(&fd, ctx->cam_w, ctx->cam_h,
                    NvBufferLayout_Pitch,
                    get_nvbuff_color_fmt(ctx->cam_pixfmt)))
            ERROR_RETURN("Failed to create NvBuffer");

        ctx->g_buff[index].dmabuff_fd = fd;

        if (-1 == NvBufferGetParams(fd, &params))
            ERROR_RETURN("Failed to get NvBuffer parameters");

        // TODO add multi-planar support
        // Currently it supports only YUV422 interlaced single-planar
        ctx->g_buff[index].size = params.height[0] * params.pitch[0];
        ctx->g_buff[index].start = (unsigned char *)mmap(
                NULL,
                ctx->g_buff[index].size,
                PROT_READ | PROT_WRITE,
                MAP_SHARED,
                ctx->g_buff[index].dmabuff_fd, 0);
    }

    if (!request_camera_buff(ctx))
        ERROR_RETURN("Failed to set up camera buff");

    if (!enqueue_vic_buff(ctx))
        ERROR_RETURN("Failed to enqueue empty buff into VIC");

    INFO("Succeed in preparing stream buffers");
    return true;
}

static bool
start_stream(context_t * ctx)
{
    enum v4l2_buf_type type;

    // Start v4l2 streaming
    type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (ioctl(ctx->cam_fd, VIDIOC_STREAMON, &type) < 0)
        ERROR_RETURN("Failed to start streaming: %s (%d)",
                strerror(errno), errno);

    // Start VIC output plane
    if (ctx->conv->output_plane.setStreamStatus(true) < 0)
        ERROR_RETURN("Failed to start VIC output plane streaming");

    // Start VIC capture plane
    if (ctx->conv->capture_plane.setStreamStatus(true) < 0)
        ERROR_RETURN("Failed to start VIC capture plane streaming");

    usleep(200);

    INFO("Camera video streaming on ...");
    return true;
}

static void
abort(context_t *ctx)
{
    ctx->got_error = true;
    if (ctx->conv)
        ctx->conv->abort();
}

static bool
conv_output_dqbuf_thread_callback(struct v4l2_buffer *v4l2_buf,
                                   NvBuffer * buffer, NvBuffer * shared_buffer,
                                   void *arg)
{
    context_t *ctx = (context_t *) arg;
    nv_buffer * cam_g_buff;

    if (!v4l2_buf)
    {
        abort(ctx);
        ERROR_RETURN("Failed to dequeue conv output plane buffer");
    }

    // Fetch nv_buffer to do format conversion
    pthread_mutex_lock(&ctx->queue_lock);
    while (ctx->conv_output_plane_buf_queue->empty())
    {
        pthread_cond_wait(&ctx->queue_cond, &ctx->queue_lock);
    }
    cam_g_buff = ctx->conv_output_plane_buf_queue->front();
    ctx->conv_output_plane_buf_queue->pop();
    pthread_mutex_unlock(&ctx->queue_lock);

    // Got EOS signal and return
    if (cam_g_buff->dmabuff_fd == 0)
        return false;
    else
    {
        // Enqueue vic output plane
        v4l2_buf->m.planes[0].m.fd =
            (unsigned long)cam_g_buff->dmabuff_fd;
        v4l2_buf->m.planes[0].bytesused = cam_g_buff->size;
    }

    if (ctx->conv->output_plane.qBuffer(*v4l2_buf, NULL) < 0)
    {
        abort(ctx);
        ERROR_RETURN("Failed to enqueue VIC output plane");
    }

    return true;
}

static bool
conv_capture_dqbuf_thread_callback(struct v4l2_buffer *v4l2_buf,
                                   NvBuffer * buffer, NvBuffer * shared_buffer,
                                   void *arg)
{
    context_t *ctx = (context_t *) arg;

    if (ctx->enable_cuda)
    {
        // Create EGLImage from dmabuf fd
        ctx->egl_image = NvEGLImageFromFd(ctx->egl_display, buffer->planes[0].fd);
        if (ctx->egl_image == NULL)
            ERROR_RETURN("Failed to map dmabuf fd (0x%X) to EGLImage",
                    buffer->planes[0].fd);

        // Running algo process with EGLImage via GPU multi cores
        HandleEGLImage(&ctx->egl_image);

        // Destroy EGLImage
        NvDestroyEGLImage(ctx->egl_display, ctx->egl_image);
        ctx->egl_image = NULL;
    }

    // Render the frame into display
    if (v4l2_buf->m.planes[0].bytesused)
        ctx->renderer->render(buffer->planes[0].fd);

    if (ctx->conv->capture_plane.qBuffer(*v4l2_buf, buffer) < 0)
    {
        abort(ctx);
        ERROR_RETURN("Failed to queue buffer on VIC capture plane");
    }

    return true;
}

static void
signal_handle(int signum)
{
    printf("Quit due to exit command from user!\n");
    quit = true;
}

static bool
start_capture(context_t * ctx)
{
    struct sigaction sig_action;
    struct pollfd fds[1];

    // Ensure a clean shutdown if user types <ctrl+c>
    sig_action.sa_handler = signal_handle;
    sigemptyset(&sig_action.sa_mask);
    sig_action.sa_flags = 0;
    sigaction(SIGINT, &sig_action, NULL);

    ctx->conv->capture_plane.setDQThreadCallback(conv_capture_dqbuf_thread_callback);
    ctx->conv->output_plane.setDQThreadCallback(conv_output_dqbuf_thread_callback);

    // Start VIC processing thread
    ctx->conv->capture_plane.startDQThread(ctx);
    ctx->conv->output_plane.startDQThread(ctx);

    // Enable render profiling information
    ctx->renderer->enableProfiling();

    fds[0].fd = ctx->cam_fd;
    fds[0].events = POLLIN;
    while (poll(fds, 1, 5000) > 0 && !ctx->got_error &&
            !ctx->conv->isInError() && !quit)
    {
        if (fds[0].revents & POLLIN) {
            struct v4l2_buffer v4l2_buf;

            // Dequeue camera buff
            memset(&v4l2_buf, 0, sizeof(v4l2_buf));
            v4l2_buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            v4l2_buf.memory = V4L2_MEMORY_DMABUF;
            if (ioctl(ctx->cam_fd, VIDIOC_DQBUF, &v4l2_buf) < 0)
                ERROR_RETURN("Failed to dequeue camera buff: %s (%d)",
                        strerror(errno), errno);

            ctx->frame++;

            if (ctx->frame == ctx->save_n_frame)
                save_frame_to_file(ctx, &v4l2_buf);

            // Push nv_buffer into conv output queue for conversion
            pthread_mutex_lock(&ctx->queue_lock);
            ctx->conv_output_plane_buf_queue->push(&ctx->g_buff[v4l2_buf.index]);
            pthread_cond_broadcast(&ctx->queue_cond);
            pthread_mutex_unlock(&ctx->queue_lock);

            // Enqueue camera buff
            // It might be more reasonable to wait for the completion of
            // VIC processing before enqueue current buff. But VIC processing
            // time is far less than camera frame interval, so we probably
            // don't need such synchonization.
            if (ioctl(ctx->cam_fd, VIDIOC_QBUF, &v4l2_buf))
                ERROR_RETURN("Failed to queue camera buffers: %s (%d)",
                        strerror(errno), errno);
        }
    }

    if (quit && !ctx->conv->isInError())
    {
        // Signal EOS to the dq thread of VIC output plane
        ctx->g_buff[0].dmabuff_fd = 0;

        pthread_mutex_lock(&ctx->queue_lock);
        ctx->conv_output_plane_buf_queue->push(&ctx->g_buff[0]);
        pthread_cond_broadcast(&ctx->queue_cond);
        pthread_mutex_unlock(&ctx->queue_lock);
    }

    // Stop VIC dq thread
    if (!ctx->got_error)
    {
        ctx->conv->waitForIdle(2000);
        ctx->conv->capture_plane.stopDQThread();
        ctx->conv->output_plane.stopDQThread();
    }

    // Print profiling information when streaming stops.
    ctx->renderer->printProfilingStats();

    return true;
}

static bool
stop_stream(context_t * ctx)
{
    enum v4l2_buf_type type;

    // Stop v4l2 streaming
    type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (ioctl(ctx->cam_fd, VIDIOC_STREAMOFF, &type))
        ERROR_RETURN("Failed to stop streaming: %s (%d)",
                strerror(errno), errno);

    // Stop VIC output plane
    if (ctx->conv->output_plane.setStreamStatus(false) < 0)
        ERROR_RETURN("Failed to stop output plane streaming");

    // Stop VIC capture plane
    if (ctx->conv->capture_plane.setStreamStatus(false) < 0)
        ERROR_RETURN("Failed to stop output plane streaming");

    INFO("Camera video streaming off ...");
    return true;
}

int
main(int argc, char *argv[])
{
    context_t ctx;
    int error = 0;

    set_defaults(&ctx);

    CHECK_ERROR(parse_cmdline(&ctx, argc, argv), cleanup,
            "Invalid options specified");

    CHECK_ERROR(init_components(&ctx), cleanup,
            "Failed to initialize v4l2 components");

    CHECK_ERROR(prepare_buffers(&ctx), cleanup,
            "Failed to prepare v4l2 buffs");

    CHECK_ERROR(start_stream(&ctx), cleanup,
            "Failed to start streaming");

    CHECK_ERROR(start_capture(&ctx), cleanup,
            "Failed to start capturing")

    CHECK_ERROR(stop_stream(&ctx), cleanup,
            "Failed to stop streaming");

cleanup:
    if (ctx.cam_fd > 0)
        close(ctx.cam_fd);

    if (ctx.renderer != NULL)
        delete ctx.renderer;

    if (ctx.egl_display && !eglTerminate(ctx.egl_display))
        printf("Failed to terminate EGL display connection\n");

    if (ctx.conv != NULL)
    {
        if (ctx.conv->isInError())
        {
            printf("Video converter is in error\n");
            error = 1;
        }
        delete ctx.conv;
    }

    if (ctx.g_buff != NULL)
    {
        for (unsigned i = 0; i < V4L2_BUFFERS_NUM; i++)
            if (ctx.g_buff[i].dmabuff_fd)
                NvBufferDestroy(ctx.g_buff[i].dmabuff_fd);
        free(ctx.g_buff);
    }

    delete ctx.conv_output_plane_buf_queue;

    if (error)
        printf("App run failed\n");
    else
        printf("App run was successful\n");

    return -error;
}
