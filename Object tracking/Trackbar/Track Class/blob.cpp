#include "blob.hpp"

#include <cmath>

Blob::Blob(int x, unsigned long int time) {
    id = x;
    create_time = time;
    update_time = 0;
}

bool Blob::is_removed(unsigned long int time) {
    if(update_time == 0 && create_time == time)
        return false;
    
    long long int diff = time - update_time;
    if(diff > 10) return true;

    return false;
}

unsigned long int Blob::get_life() {
    return (update_time - create_time);
}

void Blob::set_update_time(unsigned long int _update_time) {
    update_time =_update_time;
} 