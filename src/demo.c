#define _GNU_SOURCE
#include "network.h"
#include "detection_layer.h"
#include "region_layer.h"
#include "cost_layer.h"
#include "utils.h"
#include "parser.h"
#include "box.h"
#include "image.h"
#include "demo.h"
#include "darknet.h"

#ifdef WIN32
#include <time.h>
#include "gettimeofday.h"
#else
#include <sys/time.h>
#endif

#ifdef V4L2
#include "v4l2.h"
#endif

#ifdef OPENCV

#include "http_stream.h"

/*
static char **demo_names;
static image **demo_alphabet;
static int demo_classes;

static int nboxes = 0;
static detection *dets = NULL;

static network net;
static image in_s ;
static image det_s;


static cap_cv *cap;
static float fps = 0;
static float demo_thresh = 0;
static int demo_ext_output = 0;
static long long int frame_id = 0;
static int demo_json_port = -1;
static bool demo_skip_frame = false;


static int avg_frames;
static int demo_index = 0;
static mat_cv** cv_images;

mat_cv* in_img;
mat_cv* det_img;
mat_cv* show_img;

static volatile int flag_exit;
static int letter_box = 0;
*/

static cap_cv *cap;
static mat_cv** cv_images;

mat_cv* in_img;
mat_cv* det_img;
mat_cv* show_img;

static int avg_frames;
static const int thread_wait_ms = 1;
static volatile int run_fetch_in_thread = 0;
static volatile int run_detect_in_thread = 0;
static bool demo_skip_frame = false;

//=========================================
extern int buff_index=0;
extern int cnt = 0;
extern double cycle_array[QLEN] = {0,};
extern int ondemand = 1;
extern int num_object = 0;
extern int measure = 1;

extern int nboxes = 0;
extern detection *dets = NULL;

extern float fps = 0;
extern float demo_thresh = 0;
extern int demo_ext_output = 0;
extern long long int frame_id = 0;
extern int demo_json_port = -1;
extern int demo_index = 0;
extern int letter_box = 0;
extern int fetch_offset = 0; // zero slack

extern double e_fetch_sum = 0;
extern double b_fetch_sum = 0;
extern double d_fetch_sum = 0;
extern double e_infer_cpu_sum = 0;
extern double e_infer_gpu_sum = 5;
extern double d_infer_sum = 0;
extern double e_disp_sum = 0;
extern double b_disp_sum = 0;
extern double d_disp_sum = 0;
extern double slack_sum = 0;
extern double e2e_delay_sum = 0;
extern double fps_sum = 0;
extern double cycle_time_sum = 0;
extern double inter_frame_gap_sum = 0;
extern double num_object_sum = 0;
extern double trace_data_sum = 0;

int *fd_handler = NULL;

/* Save result in csv*/
int write_result(void)
{
    static int exist=0;
    FILE *fp;
    char file_path[256] = "";
    int tick = 0;

    strcat(file_path, MEASUREMENT_PATH);
    strcat(file_path, MEASUREMENT_FILE);

    fp=fopen(file_path,"w+");

    if (fp == NULL) 
    {
        /* make directory */
        while(!exist)
        {
            int result;

            usleep(10 * 1000);

            result = mkdir(MEASUREMENT_PATH, 0766);
            if(result == 0) { 
                exist = 1;

                fp=fopen(file_path,"w+");
            }

            if(tick == 100)
            {
                fprintf(stderr, "\nERROR: Fail to Create %s\n", file_path);

                return -1;
            }
            else tick++;
        }
    }
    else printf("\nWrite output in %s\n", file_path); 

    fprintf(fp, "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n", "e_fetch", "b_fetch", "d_fetch",
            "e_infer", "b_infer", "d_infer", "e_disp", "b_disp", "d_disp",
            "slack", "e2e_delay", "fps", "c_sys", "IFG", "n_obj");

    for(int i=0;i<OBJ_DET_CYCLE_IDX;i++)
    {
        e_fetch_sum += e_fetch_array[i];
        b_fetch_sum += b_fetch_array[i];
        d_fetch_sum += d_fetch_array[i];
        e_infer_cpu_sum += e_infer_cpu_array[i];
        e_infer_gpu_sum += e_infer_gpu_array[i];
        d_infer_sum += d_infer_array[i];
        e_disp_sum += e_disp_array[i];
        b_disp_sum += b_disp_array[i];
        d_disp_sum += d_disp_array[i];
        slack_sum += slack[i];
        e2e_delay_sum += e2e_delay[i];
        fps_sum += fps_array[i];
        cycle_time_sum += cycle_time_array[i];
        inter_frame_gap_sum += (double)inter_frame_gap_array[i];
        num_object_sum += (double)num_object_array[i];

        fprintf(fp, "%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%d,%d\n", e_fetch_array[i], b_fetch_array[i],d_fetch_array[i], 
                e_infer_cpu_array[i], e_infer_gpu_array[i], d_infer_array[i], e_disp_array[i], b_disp_array[i], d_disp_array[i], 
                slack[i], e2e_delay[i], fps_array[i], cycle_time_array[i], inter_frame_gap_array[i], num_object_array[i]);
    }
    fclose(fp);

    return 1;
}

void push_data(void)
{
    b_fetch_array[cnt - CYCLE_OFFSET] = b_fetch;
    e_fetch_array[cnt - CYCLE_OFFSET] = d_fetch - b_fetch - fetch_offset;
    //e_fetch_array[cnt - CYCLE_OFFSET] = d_fetch - b_fetch - fetch_offset;
    d_fetch_array[cnt - CYCLE_OFFSET] = d_fetch;
    inter_frame_gap_array[cnt - CYCLE_OFFSET] = inter_frame_gap;
    transfer_delay_array[cnt - CYCLE_OFFSET] = transfer_delay;

    e_infer_cpu_array[cnt - CYCLE_OFFSET] = d_infer - e_infer_gpu;
    e_infer_gpu_array[cnt - CYCLE_OFFSET] = e_infer_gpu;
    d_infer_array[cnt - CYCLE_OFFSET] = d_infer;

    fps_array[cnt - CYCLE_OFFSET] = fps;
    cycle_time_array[cnt - CYCLE_OFFSET] = 1000./fps;
    e2e_delay[cnt - CYCLE_OFFSET] = end_disp - frame[display_index].frame_timestamp;
    printf("end_disp : %f \n", end_disp);    
    printf("timestamp : %f \n", frame[display_index].frame_timestamp);
    //printf("e2e: %f\n",end_disp - frame[display_index].frame_timestamp);
#ifdef V4L2
    printf("e2e: %f\n",end_disp - frame[display_index].frame_timestamp);
#endif
    e_disp_array[cnt - CYCLE_OFFSET] = d_disp - b_disp;
    b_disp_array[cnt - CYCLE_OFFSET] = b_disp;
    d_disp_array[cnt - CYCLE_OFFSET] = d_disp;
    slack[cnt - CYCLE_OFFSET] = slack_time;
    num_object_array[cnt - CYCLE_OFFSET] = num_object;

    //printf("num_object : %d\n", num_object);
    //printf("slack: %f\n",slack[cnt-CYCLE_OFFSET]);
    printf("latency: %f\n",e2e_delay[cnt - CYCLE_OFFSET]);
    printf("cnt : %d\n",cnt);

    return;
}

/* Timestamp in ms */
double get_time_in_ms(void)
{
    struct timespec time_after_boot;
    clock_gettime(CLOCK_MONOTONIC,&time_after_boot);
    return (time_after_boot.tv_sec*1000+time_after_boot.tv_nsec*0.000001);
}


void *fetch_in_thread(void *ptr)
{
    //start_fetch = get_time_in_ms();

    while (!custom_atomic_load_int(&flag_exit)) {
        while (!custom_atomic_load_int(&run_fetch_in_thread)) {
            if (custom_atomic_load_int(&flag_exit)) return 0;
            if (demo_skip_frame)
                consume_frame(cap);
            this_thread_yield();
        }

        start_fetch = get_time_in_ms(); 
        int dont_close_stream = 0;    // set 1 if your IP-camera periodically turns off and turns on video-stream
        if (letter_box)
            in_s = get_image_from_stream_letterbox(cap, net.w, net.h, net.c, &in_img, dont_close_stream);
        else{
#ifdef V4L2
		    if(-1 == capture_image(&frame[buff_index], *fd_handler))
		    {
			    perror("Fail to capture image");
			    exit(0);
		    }

            letterbox_image_into(frame[buff_index].frame, net.w, net.h, frame[buff_index].resize_frame);
            //frame[buff_index].resize_frame = letterbox_image(frame[buff_index].frame, net.w, net.h);
            //show_image_cv(frame[buff_index].resize_frame,"im");

            if(!frame[buff_index].resize_frame.data)
            {
                printf("Stream closed.\n");
                flag_exit = 1;
                //exit(EXIT_FAILURE);
                return 0;
            }
#else
            in_s = get_image_from_stream_resize_with_timestamp(cap, net.w, net.h, net.c, &in_img, dont_close_stream, &frame[buff_index]);
            //in_s = get_image_from_stream_resize(cap, net.w, net.h, net.c, &in_img, dont_close_stream);
            if (!in_s.data) 
            {
                printf("Stream closed.\n");
                custom_atomic_store_int(&flag_exit, 1);
                custom_atomic_store_int(&run_fetch_in_thread, 0);
                //exit(EXIT_FAILURE);
                return 0;
            }
#endif
            //in_s = resize_image(in, net.w, net.h);

            custom_atomic_store_int(&run_fetch_in_thread, 0);

            end_fetch = get_time_in_ms();

   		    inter_frame_gap = GET_IFG(frame[buff_index].frame_sequence, frame_sequence_tmp);

            if(cnt >= (CYCLE_OFFSET - 5))
            {
                d_fetch = end_fetch - start_fetch;
                b_fetch = frame[buff_index].select;
                e_fetch = d_fetch - b_fetch - fetch_offset;
            }   

        }
    }

     /*   
    end_fetch = get_time_in_ms();
    
    if(cnt >= (CYCLE_OFFSET - 5))
    {
        d_fetch = end_fetch - start_fetch;
        b_fetch = frame[buff_index].select;
        e_fetch = d_fetch - b_fetch - fetch_offset;
    }
    */
    return 0;
}

void *fetch_in_thread_sync(void *ptr)
{
    custom_atomic_store_int(&run_fetch_in_thread, 1);
    while (custom_atomic_load_int(&run_fetch_in_thread)) this_thread_sleep_for(thread_wait_ms);
    return 0;
}

void *detect_in_thread(void *ptr)
{
    //start_infer = get_time_in_ms();
    while (!custom_atomic_load_int(&flag_exit)) {
        while (!custom_atomic_load_int(&run_detect_in_thread)) {
            //start_infer = get_time_in_ms();
            if (custom_atomic_load_int(&flag_exit)) return 0;
            this_thread_yield();
        }

        start_infer = get_time_in_ms();

        layer l = net.layers[net.n - 1];
#ifdef V4L2
        float *X = frame[detect_index].resize_frame.data;
#else
        float *X = det_s.data;
#endif
        //float *prediction =
        network_predict(net, X);
	
        cv_images[demo_index] = det_img;
        det_img = cv_images[(demo_index + avg_frames / 2 + 1) % avg_frames];
        demo_index = (demo_index + 1) % avg_frames;
/*
#ifdef V4L2
    dets = get_network_boxes(&net, net.w, net.h, demo_thresh, demo_thresh, 0, 1, &nboxes, 0); // resized
#else
        if (letter_box)
            dets = get_network_boxes(&net, get_width_mat(in_img), get_height_mat(in_img), demo_thresh, demo_thresh, 0, 1, &nboxes, 1); // letter box
        else
            dets = get_network_boxes(&net, net.w, net.h, demo_thresh, demo_thresh, 0, 1, &nboxes, 0); // resized

#endif
*/

        if (letter_box)
            dets = get_network_boxes(&net, get_width_mat(in_img), get_height_mat(in_img), demo_thresh, demo_thresh, 0, 1, &nboxes, 1); // letter box
        else
            dets = get_network_boxes(&net, net.w, net.h, demo_thresh, demo_thresh, 0, 1, &nboxes, 0); // resized

        //const float nms = .45;
        //if (nms) {
        //    if (l.nms_kind == DEFAULT_NMS) do_nms_sort(dets, nboxes, l.classes, nms);
        //    else diounms_sort(dets, nboxes, l.classes, nms, l.nms_kind, l.beta_nms);
        //}

        custom_atomic_store_int(&run_detect_in_thread, 0);

        end_infer = get_time_in_ms();

        d_infer = end_infer - start_infer;
    }
/*
    end_infer = get_time_in_ms();

    d_infer = end_infer - start_infer;
    return 0;
*/
}

void *detect_in_thread_sync(void *ptr)
{
    custom_atomic_store_int(&run_detect_in_thread, 1);
    while (custom_atomic_load_int(&run_detect_in_thread)) this_thread_sleep_for(thread_wait_ms);
    return 0;
}

double get_wall_time()
{
    struct timeval walltime;
    if (gettimeofday(&walltime, NULL)) {
        return 0;
    }
    return (double)walltime.tv_sec + (double)walltime.tv_usec * .000001;
}

void *display_thread(void *ptr)
{
    int c = show_image_cv(frame[display_index].frame, "Demo");

    if (c == 27 || c == 1048603) // ESC - exit (OpenCV 2.x / 3.x)
    {
        flag_exit = 1;
    }

    return 0;
}

void demo(char *cfgfile, char *weightfile, float thresh, float hier_thresh, int cam_index, const char *filename, char **names, int classes, int avgframes,
    int frame_skip, char *prefix, char *out_filename, int mjpeg_port, int dontdraw_bbox, int json_port, int dont_show, int ext_output, int letter_box_in, int time_limit_sec, char *http_post_host,
    int benchmark, int benchmark_layers, int w, int h, int cam_fps)
{
    if (avgframes < 1) avgframes = 1;
    avg_frames = avgframes;
    letter_box = letter_box_in;
    in_img = det_img = show_img = NULL;
    //skip = frame_skip;
    image **alphabet = load_alphabet();
    int delay = frame_skip;
    demo_names = names;
    demo_alphabet = alphabet;
    demo_classes = classes;
    demo_thresh = thresh;
    demo_ext_output = ext_output;
    demo_json_port = json_port;
    printf("Demo\n");
    net = parse_network_cfg_custom(cfgfile, 1, 1);    // set batch=1
    if(weightfile){
        load_weights(&net, weightfile);
    }
    if (net.letter_box) letter_box = 1;
    net.benchmark_layers = benchmark_layers;
    fuse_conv_batchnorm(net);
    calculate_binary_weights(net);
    srand(2222222);

    int img_w = w;
    int img_h = h;
    int cam_frame_rate = cam_fps;
    char *pipeline = NULL;

    if(filename){
        printf("video file: %s\n", filename);
        cap = get_capture_video_stream(filename);
        demo_skip_frame = is_live_stream(filename);
    }else{
        printf("Webcam index: %d\n", cam_index);
        printf(" %d, %d \n", img_w, img_h);
#ifdef V4L2
        char cam_dev[256] = "/dev/video";
        char index[256];
        sprintf(index, "%d", cam_index);
        strcat(cam_dev, index);
        printf("cam dev : %s\n", cam_dev);

        fd_handler = open_device(cam_dev, cam_frame_rate, img_w, img_h);
        if(fd_handler ==  NULL)
        {
            perror("Couldn't connect to webcam.\n");
        }
#else
        cap = get_capture_webcam(cam_index);
        demo_skip_frame = true;
    }

    if (!cap) {
#ifdef WIN32
        printf("Check that you have copied file opencv_ffmpeg340_64.dll to the same directory where is darknet.exe \n");
#endif
        error("Couldn't connect to webcam.", DARKNET_LOC);
    }
#endif

    layer l = net.layers[net.n-1];
    int j;

    cv_images = (mat_cv**)xcalloc(avg_frames, sizeof(mat_cv));

    int i;
    for (i = 0; i < net.n; ++i) {
        layer lc = net.layers[i];
        if (lc.type == YOLO) {
            lc.mean_alpha = 1.0 / avg_frames;
            l = lc;
        }
    }

    if (l.classes != demo_classes) {
        printf("\n Parameters don't match: in cfg-file classes=%d, in data-file classes=%d \n", l.classes, demo_classes);
        getchar();
        exit(0);
    }

    flag_exit = 0;

    custom_thread_t fetch_thread = NULL;
    custom_thread_t detect_thread = NULL;
    if (custom_create_thread(&fetch_thread, 0, fetch_in_thread, 0)) error("Thread creation failed", DARKNET_LOC);
    if (custom_create_thread(&detect_thread, 0, detect_in_thread, 0)) error("Thread creation failed", DARKNET_LOC);

#ifdef V4L2
	if(-1 == capture_image(&frame[buff_index], *fd_handler))
	{
		perror("Fail to capture image");
		exit(0);
	}
    frame[0].resize_frame = letterbox_image(frame[0].frame, net.w, net.h);

    frame[1].frame = frame[0].frame;
    frame[1].resize_frame = letterbox_image(frame[0].frame, net.w, net.h);

    frame[2].frame = frame[0].frame;
    frame[2].resize_frame = letterbox_image(frame[0].frame, net.w, net.h);
#else
    fetch_in_thread_sync(0); //fetch_in_thread(0);
    det_img = in_img;
    det_s = in_s;

    fetch_in_thread_sync(0); //fetch_in_thread(0);
    detect_in_thread_sync(0); //fetch_in_thread(0);
    det_img = in_img;
    det_s = in_s;

    for (j = 0; j < avg_frames / 2; ++j) {
        free_detections(dets, nboxes);
        fetch_in_thread_sync(0); //fetch_in_thread(0);
        detect_in_thread_sync(0); //fetch_in_thread(0);
        det_img = in_img;
        det_s = in_s;
    }
#endif

    int count = 0;
    if(!prefix && !dont_show){
        int full_screen = 0;
        create_window_cv("Demo", full_screen, 1352, 1013);
    }


    write_cv* output_video_writer = NULL;
    if (out_filename && !flag_exit)
    {
        int src_fps = 25;
        src_fps = get_stream_fps_cpp_cv(cap);

        output_video_writer =
            create_video_writer(out_filename, 'D', 'I', 'V', 'X', src_fps, get_width_mat(det_img), get_height_mat(det_img), 1);

        //'H', '2', '6', '4'
        //'D', 'I', 'V', 'X'
        //'M', 'J', 'P', 'G'
        //'M', 'P', '4', 'V'
        //'M', 'P', '4', '2'
        //'X', 'V', 'I', 'D'
        //'W', 'M', 'V', '2'
    }

    int send_http_post_once = 0;
    const double start_time_lim = get_time_point();
    double before = get_time_point();
    double start_time = get_time_point();
    float avg_fps = 0;
    int frame_counter = 0;
    int global_frame_counter = 0;

    while(1){
        ++count;
        {

            display_index = (buff_index + 1) %3;
            detect_index = (buff_index + 2) %3;

            const float nms = .45;    // 0.4F
            int local_nboxes = nboxes;
            detection *local_dets = dets;
            this_thread_yield();

            if (!benchmark) custom_atomic_store_int(&run_fetch_in_thread, 1); // if (custom_create_thread(&fetch_thread, 0, fetch_in_thread, 0)) error("Thread creation failed", DARKNET_LOC);
            custom_atomic_store_int(&run_detect_in_thread, 1); // if (custom_create_thread(&detect_thread, 0, detect_in_thread, 0)) error("Thread creation failed", DARKNET_LOC);

            double start_disp = get_time_in_ms();

            //if (nms) do_nms_obj(local_dets, local_nboxes, l.classes, nms);    // bad results
            if (nms) {
                if (l.nms_kind == DEFAULT_NMS) do_nms_sort(local_dets, local_nboxes, l.classes, nms);
                else diounms_sort(local_dets, local_nboxes, l.classes, nms, l.nms_kind, l.beta_nms);
            }

            if (l.embedding_size) set_track_id(local_dets, local_nboxes, demo_thresh, l.sim_thresh, l.track_ciou_norm, l.track_history_size, l.dets_for_track, l.dets_for_show);

            //printf("\033[2J");
            //printf("\033[1;1H");
            //printf("\nFPS:%.1f\n", fps);
            printf("Objects:\n\n");

#ifdef V4L2
            if (!benchmark) draw_detections_v3(frame[display_index].frame, local_dets, local_nboxes, demo_thresh, demo_names, demo_alphabet, demo_classes, demo_ext_output);
            free_detections(local_dets, local_nboxes);

            draw_bbox_time = get_time_in_ms() - start_disp;

            /* Image display */
            display_thread(0);
#else
            ++frame_id;
            if (demo_json_port > 0) {
                int timeout = 400000;
                send_json(local_dets, local_nboxes, l.classes, demo_names, frame_id, demo_json_port, timeout);
            }

            //char *http_post_server = "webhook.site/898bbd9b-0ddd-49cf-b81d-1f56be98d870";
            if (http_post_host && !send_http_post_once) {
                int timeout = 3;            // 3 seconds
                int http_post_port = 80;    // 443 https, 80 http
                if (send_http_post_request(http_post_host, http_post_port, filename,
                    local_dets, nboxes, classes, names, frame_id, ext_output, timeout))
                {
                    if (time_limit_sec > 0) send_http_post_once = 1;
                }
            }

            if (!benchmark && !dontdraw_bbox) draw_detections_cv_v3(show_img, local_dets, local_nboxes, demo_thresh, demo_names, demo_alphabet, demo_classes, demo_ext_output);
            free_detections(local_dets, local_nboxes);

            draw_bbox_time = get_time_in_ms() - start_disp;

            display_thread(0);

            printf("\nFPS:%.1f \t AVG_FPS:%.1f\n", fps, avg_fps);

            if(!prefix){
                if (!dont_show) {
                    const int each_frame = max_val_cmp(1, avg_fps / 60);
                    if(global_frame_counter % each_frame == 0) show_image_mat(show_img, "Demo");

                    waitkey_start = get_time_in_ms();
                    int c = wait_key_cv(1);
                    b_disp = get_time_in_ms() - waitkey_start;
                    if (c == 10) {
                        if (frame_skip == 0) frame_skip = 60;
                        else if (frame_skip == 4) frame_skip = 0;
                        else if (frame_skip == 60) frame_skip = 4;
                        else frame_skip = 0;
                    }
                    else if (c == 27 || c == 1048603) // ESC - exit (OpenCV 2.x / 3.x)
                    {
                        flag_exit = 1;
                    }
                }
            }else{
                char buff[256];
                sprintf(buff, "%s_%08d.jpg", prefix, count);
                if(show_img) save_cv_jpg(show_img, buff);
            }

            // if you run it with param -mjpeg_port 8090  then open URL in your web-browser: http://localhost:8090
            if (mjpeg_port > 0 && show_img) {
                int port = mjpeg_port;
                int timeout = 400000;
                int jpeg_quality = 40;    // 1 - 100
                send_mjpeg(show_img, port, timeout, jpeg_quality);
            }

            // save video file
            if (output_video_writer && show_img) {
                write_frame_cv(output_video_writer, show_img);
                printf("\n cvWriteFrame \n");
            }

#endif

            /* display end */

            end_disp = get_time_in_ms();

            d_disp = end_disp - start_disp; 

            while (custom_atomic_load_int(&run_detect_in_thread)) {
                if(avg_fps > 180) this_thread_yield();
                else this_thread_sleep_for(thread_wait_ms);   // custom_join(detect_thread, 0);
            }
            if (!benchmark) {
                while (custom_atomic_load_int(&run_fetch_in_thread)) {
                    if (avg_fps > 180) this_thread_yield();
                    else this_thread_sleep_for(thread_wait_ms);   // custom_join(fetch_thread, 0);
                }
                free_image(det_s);
            }

            if (time_limit_sec > 0 && (get_time_point() - start_time_lim)/1000000 > time_limit_sec) {
                printf(" start_time_lim = %f, get_time_point() = %f, time spent = %f \n", start_time_lim, get_time_point(), get_time_point() - start_time_lim);
                break;
            }

            if (flag_exit == 1) break;

            if(delay == 0){
                if(!benchmark) release_mat(&show_img);
                show_img = det_img;
            }
            det_img = in_img;
            det_s = in_s;
        }
        --delay;
        if(delay < 0){
            delay = frame_skip;

            //double after = get_wall_time();
            //float curr = 1./(after - before);
            double after = get_time_point();    // more accurate time measurements
            float curr = 1000000. / (after - before);
            fps = fps*0.9 + curr*0.1;
            before = after;

            float spent_time = (get_time_point() - start_time) / 1000000;
            frame_counter++;
            global_frame_counter++;
            if (spent_time >= 3.0f) {
                //printf(" spent_time = %f \n", spent_time);
                avg_fps = frame_counter / spent_time;
                frame_counter = 0;
                start_time = get_time_point();
            }
        }

        cycle_array[cycle_index] = 1000./fps;
        cycle_index = (cycle_index + 1) % 4;
        //slack_time = (MAX(d_infer, d_disp)) - (d_fetch);

#ifdef MEASUREMENT
        if (cnt >= CYCLE_OFFSET) push_data();

        /* Exit object detection cycle */
        if(cnt == ((OBJ_DET_CYCLE_IDX + CYCLE_OFFSET) - 1)) 
        {
            if(-1 == write_result())
            {
                /* return error */
                exit(0);
            }

            /* exit loop */
            break;
        }
#endif

        /* Increase count */
        if(cnt != ((OBJ_DET_CYCLE_IDX + CYCLE_OFFSET)-1)) cnt++;
        /* Change buffer index */
        buff_index = (buff_index + 1) % 3;
    }

#ifdef MEASUREMENT
    /* Average data */
    printf("============ Darknet data ============\n");
    printf("Avg fetch execution time (ms) : %0.2f\n", e_fetch_sum / OBJ_DET_CYCLE_IDX);
    printf("Avg fetch blocking time (ms) : %0.2f\n", b_fetch_sum / OBJ_DET_CYCLE_IDX);
    printf("Avg fetch delay (ms) : %0.2f\n", d_fetch_sum / OBJ_DET_CYCLE_IDX);
    printf("Avg infer execution on cpu (ms) : %0.2f\n", e_infer_cpu_sum / OBJ_DET_CYCLE_IDX);
    printf("Avg infer execution on gpu (ms) : %0.2f\n", e_infer_gpu_sum / OBJ_DET_CYCLE_IDX);
    printf("Avg infer delay (ms) : %0.2f\n", d_infer_sum / OBJ_DET_CYCLE_IDX);
    printf("Avg disp execution time (ms) : %0.2f\n", e_disp_sum / OBJ_DET_CYCLE_IDX);
    printf("Avg disp blocking time (ms) : %0.2f\n", b_disp_sum / OBJ_DET_CYCLE_IDX);
    printf("Avg disp delay (ms) : %0.2f\n", d_disp_sum / OBJ_DET_CYCLE_IDX);
    printf("Avg salck (ms) : %0.2f\n", slack_sum / OBJ_DET_CYCLE_IDX);
    printf("Avg E2E delay (ms) : %0.2f\n", e2e_delay_sum / OBJ_DET_CYCLE_IDX);
    printf("Avg cycle time (ms) : %0.2f\n", cycle_time_sum / OBJ_DET_CYCLE_IDX);
    printf("Avg inter frame gap : %0.2f\n", inter_frame_gap_sum / OBJ_DET_CYCLE_IDX);
    printf("Avg number of object : %0.2f\n", num_object_sum / OBJ_DET_CYCLE_IDX);
    printf("=====================================\n");
#endif

    printf("input video stream closed. \n");
    if (output_video_writer) {
        release_video_writer(&output_video_writer);
        printf("output_video_writer closed. \n");
    }

    this_thread_sleep_for(thread_wait_ms);

    custom_join(detect_thread, 0);
    custom_join(fetch_thread, 0);

    // free memory
    free_image(in_s);
    free_detections(dets, nboxes);

    demo_index = (avg_frames + demo_index - 1) % avg_frames;
    for (j = 0; j < avg_frames; ++j) {
            release_mat(&cv_images[j]);
    }
    free(cv_images);

    free_ptrs((void **)names, net.layers[net.n - 1].classes);

    const int nsize = 8;
    for (j = 0; j < nsize; ++j) {
        for (i = 32; i < 127; ++i) {
            free_image(alphabet[j][i]);
        }
        free(alphabet[j]);
    }
    free(alphabet);
    free_network(net);
    //cudaProfilerStop();
}
}
#else
void demo(char *cfgfile, char *weightfile, float thresh, float hier_thresh, int cam_index, const char *filename, char **names, int classes, int avgframes,
    int frame_skip, char *prefix, char *out_filename, int mjpeg_port, int dontdraw_bbox, int json_port, int dont_show, int ext_output, int letter_box_in, int time_limit_sec, char *http_post_host,
    int benchmark, int benchmark_layers)
{
    fprintf(stderr, "Demo needs OpenCV for webcam images.\n");
}
#endif
