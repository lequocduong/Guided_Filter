#include <chrono>
#include <iostream>
#include <dirent.h>
#include <opencv2/opencv.hpp>
#include "/mnt/c/Users/Admin/Desktop/Puwell/work/ncnn/build/install/include/ncnn/net.h"

class VideoIO
{
public:
    // Read a video by path with opencv and store all frames in a vector and return the fps
    void read_video(const char *path, std::vector<cv::Mat> &frames, double &fps)
    {
        std::cout << path << std::endl;
        cv::VideoCapture cap(path);
        if (!cap.isOpened())
        {
            std::cerr << "ERROR: Unable to open the video" << std::endl;
            return;
        }
        fps = cap.get(cv::CAP_PROP_FPS);
        cv::Mat frame;
        while (true)
        {
            cap >> frame;
            if (frame.empty())
            {
                break;
            }
            frames.push_back(frame.clone());
        }
        cap.release();
        std::cout << "Total number of frames captured: " << frames.size() << std::endl;
    }

    void read_image_folder(const char *path, std::vector<cv::Mat> &frames)
    {
        DIR *dir;
        struct dirent *ent;
        if ((dir = opendir(path)) != NULL)
        {
            while ((ent = readdir(dir)) != NULL)
            {
                if (ent->d_type == DT_REG)
                {
                    std::string filename = ent->d_name;
                    cv::Mat frame = cv::imread(std::string(path) + "/" + filename);
                    if (!frame.empty())
                    {
                        frames.push_back(frame.clone());
                    }
                }
            }
            closedir(dir);
        }
        else
        {
            std::cerr << "ERROR: Unable to open the directory" << std::endl;
        }
    }

    void write_image_folder(const char *output_path, std::vector<cv::Mat> &enhanced_frames)
    {
        DIR *dir = opendir(output_path);
        for (size_t i = 0; i < enhanced_frames.size(); ++i)
        {
            std::stringstream ss;
            ss << std::setw(4) << std::setfill('0') << i;
            std::string tmp(ss.str());
            std::string filename = "output_" + tmp + ".jpg";
            cv::imwrite(std::string(output_path) + "/" + filename, enhanced_frames[i]);
        }
        closedir(dir);
    }

    // Save the enhanced video to a file with fps
    void saveEnhancedVideo(const std::vector<cv::Mat> &enhanced_frames, const std::string &output_path, const double &fps)
    {
        {
            // Save the enhanced frames to a video file MP4 format
            cv::VideoWriter writer(output_path, cv::VideoWriter::fourcc('M', 'P', '4', 'V'), fps, cv::Size(enhanced_frames[0].cols, enhanced_frames[0].rows), true);
            if (!writer.isOpened())
            {
                std::cerr << "ERROR: Unable to open the output video" << std::endl;
                return;
            }
            for (size_t i = 0; i < enhanced_frames.size(); ++i)
            {
                writer.write(enhanced_frames[i]);
            }
            writer.release();
            std::cout << "Enhanced video saved to: " << output_path << std::endl;
        }
    }
};

class GuidedFilter
{
public:
    GuidedFilter(int r, float eps, float alpha, float beta) : r(r), eps(eps), alpha(alpha), beta(beta) {}

    std::pair<cv::Mat, cv::Mat> guidedFilter(const cv::Mat &I, const cv::Mat &p, int height, int width)
    {
        cv::Mat mean_I, mean_p, mean_Ip, cov_Ip, mean_II, var_I, a, b, mean_a, mean_b;

        // Step 1: Calculate mean of I and p
        cv::boxFilter(I, mean_I, CV_32F, cv::Size(r, r));
        cv::boxFilter(p, mean_p, CV_32F, cv::Size(r, r));

        // Step 2: Calculate mean of I*p and covariance
        cv::boxFilter(I.mul(p), mean_Ip, CV_32F, cv::Size(r, r));
        cov_Ip = mean_Ip - mean_I.mul(mean_p);

        // Step 3: Calculate variance of I
        cv::boxFilter(I.mul(I), mean_II, CV_32F, cv::Size(r, r));
        var_I = mean_II - mean_I.mul(mean_I);

        // Step 4: Calculate coefficients a and b
        a = cov_Ip / (var_I + eps);
        b = mean_p - a.mul(mean_I);

        // Step 5: Calculate mean of a and b
        cv::boxFilter(a, mean_a, CV_32F, cv::Size(r, r));
        cv::boxFilter(b, mean_b, CV_32F, cv::Size(r, r));

        // Step 6: Resize mean_a and mean_b to original image size
        cv::resize(mean_a, mean_a, cv::Size(width, height), 0, 0, cv::INTER_LINEAR);
        cv::resize(mean_b, mean_b, cv::Size(width, height), 0, 0, cv::INTER_LINEAR);

        // Adjust mean_a and mean_b
        mean_a = mean_a.mul(alpha);
        mean_b = mean_b.mul(beta);

        // Return the pair of results
        return std::make_pair(mean_a, mean_b);
    }

private:
    int r;     // Radius
    float eps; // Regularization parameter
    float alpha;
    float beta;
};

class VideoEnhancementPipeline
{
public:
    VideoEnhancementPipeline()
    {
        // model_param = "./models/reseffnet/reseffnet_gan_div3_input_270x480.ncnn.param";
        // model_bin = "./models/reseffnet/reseffnet_gan_div3_input_270x480.ncnn.bin";
        model_param = "./models/convnext-tiny/convnext_tiny_div12.ncnn.param";
        model_bin = "./models/convnext-tiny/convnext_tiny_div12.ncnn.bin";
        input_name = "in0";
        output_name = "out0";
        // Load ncnn model
        this->net.load_param(model_param);
        this->net.load_model(model_bin);

        this->input_width = 1920 / 12;
        this->input_height = 1080 / 12;
    }

    void enhanceVideo(const std::vector<cv::Mat> &frames, std::vector<cv::Mat> &enhanced_frames)
    {
        const float scal[] = {0.003915, 0.003915, 0.003915};
        const float scal2[] = {255, 255, 255};
        cv::Mat mean_a, mean_b;
        cv::Mat final_output;

        int skip_frame = 30;
        int r = 24;
        float eps = 0.00000001f;
        float alpha = 1.5f;
        float beta = 1.2f;

        // Initialize the GuidedFilter
        GuidedFilter gf(r, eps, alpha, beta);

        auto start = std::chrono::high_resolution_clock::now(); // Initialization
        auto end = std::chrono::high_resolution_clock::now();   // Initialization

        std::chrono::duration<double> preprocessed_time = end - end;
        std::chrono::duration<double> postprocessed_time = end - end;
        std::chrono::duration<double> inference_time = end - end;
        std::chrono::duration<double> guided_filter_time = end - end;
        std::chrono::duration<double> transformation_time = end - end;
        
        for (size_t i = 0; i < frames.size(); ++i)
        {
            if (i % skip_frame == 0) // First frame of the batch will be passed through the ncnn model
            {
                // Start preprocessing
                start = std::chrono::high_resolution_clock::now();

                // Preprocess the frames
                cv::Mat frame = frames[i];
                cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);

                // Resize the frame to the desired size
                cv::Mat resized_frame;
                cv::resize(frame, resized_frame, cv::Size(this->input_width, this->input_height));

                ncnn::Extractor extractor = this->net.create_extractor();

                // Convert the frame to ncnn Mat format
                in = ncnn::Mat::from_pixels(resized_frame.data, ncnn::Mat::PIXEL_RGB, resized_frame.cols, resized_frame.rows);

                // Normalize the input frame to the range of 0-1
                in.substract_mean_normalize(0, scal); // 0-255  -->  0-1

                // End preprocessing
                end = std::chrono::high_resolution_clock::now();
                preprocessed_time += end - start;

                // Start Inferencing
                start = std::chrono::high_resolution_clock::now();

                // Perform Inference
                extractor.input(input_name, in);
                extractor.extract(output_name, out);

                // Denormalize the output
                out.substract_mean_normalize(0, scal2);

                // Stop measuring time
                end = std::chrono::high_resolution_clock::now();

                // Calculate and print inference time
                inference_time += end - start;

                // Convert NCNN output back to OpenCV Mat
                cv::Mat ncnn_output_image(out.h, out.w, CV_8UC3);
                out.to_pixels(ncnn_output_image.data, ncnn::Mat::PIXEL_RGB);

                // Start guided filtering
                start = std::chrono::high_resolution_clock::now();

                // Convert to float
                resized_frame.convertTo(resized_frame, CV_32FC3, 1.0 / 255.0);
                frame.convertTo(frame, CV_32FC3, 1.0 / 255.0);
                ncnn_output_image.convertTo(ncnn_output_image, CV_32FC3, 1.0 / 255.0);

                // Apply the guided filter
                auto result = gf.guidedFilter(resized_frame, ncnn_output_image, frame.rows, frame.cols);
                mean_a = result.first;
                mean_b = result.second;

                // Stop measuring time
                end = std::chrono::high_resolution_clock::now();
                guided_filter_time += end - start;

                // Start Transformation
                start = std::chrono::high_resolution_clock::now();

                // Calculate the output
                final_output = mean_a.mul(frame) + mean_b;

                end = std::chrono::high_resolution_clock::now();
                transformation_time += end - start;

                // Start Postprocessing
                start = std::chrono::high_resolution_clock::now();

                // Convert the final output back to 8-bit
                final_output.convertTo(final_output, CV_8UC3, 255.0);

                // Convert RGB to BGR
                cv::cvtColor(final_output, final_output, cv::COLOR_RGB2BGR);

                // End Postprocessing
                end = std::chrono::high_resolution_clock::now();
                postprocessed_time += end - start;
            }
            else // The other frames in the batch will be applied linear transformation
            {
                // Start Preprocessing
                start = std::chrono::high_resolution_clock::now();

                cv::Mat frame = frames[i];
                cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);

                // Normalize the input frame
                frame.convertTo(frame, CV_32FC3, 1.0 / 255.0);

                // End Preprocessing
                end = std::chrono::high_resolution_clock::now();
                preprocessed_time += end - start;

                // Start Transformation
                start = std::chrono::high_resolution_clock::now();

                // Calculate the output
                final_output = mean_a.mul(frame) + mean_b;

                // End Transformation
                end = std::chrono::high_resolution_clock::now();
                transformation_time += end - start;

                // Start Postprocessing
                start = std::chrono::high_resolution_clock::now();

                // Convert the final output back to 8-bit
                final_output.convertTo(final_output, CV_8UC3, 255.0);

                // Convert RGB to BGR
                cv::cvtColor(final_output, final_output, cv::COLOR_RGB2BGR);

                // End Postprocessing
                end = std::chrono::high_resolution_clock::now();
                postprocessed_time += end - start;
            }

            // Save the enhanced frame to enhanced_frames
            enhanced_frames.push_back(final_output.clone());
        }

        // Print out the time
        std::cout << "Total Preprocessing Time: " << preprocessed_time.count() << " seconds" << std::endl;
        std::cout << "Total Inference Time: " << inference_time.count() << " seconds" << std::endl;
        std::cout << "Total Guided Filter Time: " << guided_filter_time.count() << " seconds" << std::endl;
        std::cout << "Total Transformation Time: " << transformation_time.count() << " seconds" << std::endl;
        std::cout << "Total Postprocessing Time: " << postprocessed_time.count() << " seconds" << std::endl;
        // TOTAL TIME
        std::cout << "TOTAL: " << preprocessed_time.count() + inference_time.count() + guided_filter_time.count() + transformation_time.count() + postprocessed_time.count() << " seconds" << std::endl;
    }

private:
    ncnn::Net net;
    const char *model_param;
    const char *model_bin;
    const char *input_name;
    const char *output_name;
    int input_width;
    int input_height;
    ncnn::Mat in;
    ncnn::Mat out;
};

int main()
{
    std::vector<cv::Mat> frames;
    std::vector<cv::Mat> enhanced_frames;

    // Read the video
    VideoEnhancementPipeline vep;
    VideoIO io;

    std::cout << "Read image folder" << std::endl;
    io.read_image_folder("imgs", frames);

    // Perform video enhancement
    std::cout << "START INFERENCE" << std::endl;
    vep.enhanceVideo(frames, enhanced_frames);

    // Save the enhanced video
    io.write_image_folder("output_vid", enhanced_frames);

    return 0;
}