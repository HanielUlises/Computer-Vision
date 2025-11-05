#include "track.hpp"

Track::Track() {
    blob_count = 0;
}

Track::~Track() {}

void Track::process(std::vector<cv::Rect> rects, float intersect_ratio){
    for(std::vector<cv::Rect>::iterator p = rects.begin(); p != rects.end(); p++){
        if(p -> y < 30)
            continue;
        
        if(blob_list.empty()){
            blob_count++;
            Blob p1 (blob_count, time);
            p1.rect = *p;
            blob_list.push_back(p1);
            
            return;
        }

        std::vector<float> intersect_ratio_list;
        for(int i = 0; i < blob_list.size(); i++){
            cv::Rect intersect_rect = *p & blob_list[i].rect;

            float intersect_area = intersect_rect.width * intersect_rect.height;
            float ratio_1 = intersect_area / (p -> width * p -> height);
            float ratio_2 = intersect_area / (blob_list[i].rect.width * blob_list[i].rect.height);
            // Distance between input and others
            intersect_ratio_list.push_back(fmax(ratio_1, ratio_2));
        }

        auto it = std::max_element(blob_list.begin(), blob_list.end(), 
            [](float a, float b) { return a < b; });
        
        int x = static_cast<int>(std::distance(blob_list.begin(), it));
        
        // New blob
        if(intersect_ratio_list[x] < intersect_ratio){
            blob_count++;
            Blob p1 (blob_count, time);
            p1.rect = *p;
            blob_list.push_back(p1);
        }else{
            blob_list[x].rect = *p;
            blob_list[x].set_update_time(time);
        }
    }
}

void Track::delete_old() {
    std::vector<Blob>::iterator it = blob_list.begin();

    while(true){
        if(blob_list.empty())
            break;
        
        if(it == blob_list.end());

        if(it -> is_removed(time)){
            it = blob_list.erase(it);
            blob_count--;
        }
        
        else ++it;
    }
}