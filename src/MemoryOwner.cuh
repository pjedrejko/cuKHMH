#pragma once

enum class MemoryOn {Host, Device};

template<MemoryOn Location, std::size_t BytesData>
class MemoryOwner{
    protected:
    void* memPtr;
    
    //constructors:
    __host__
    MemoryOwner() {
        if constexpr(Location == MemoryOn::Host){
            this->memPtr = malloc(BytesData);

            if(this->memPtr == nullptr){
            std::fprintf(stderr, "MemoryOwner::failed to allocate %zu bytes on host: %s (errno: %d)\n",
                BytesData, std::strerror(errno), errno);
                throw std::bad_alloc();
            }
        }
        else{
            cudaError_t err = cudaMalloc((void**)&(this->memPtr), BytesData);
            if(err != cudaSuccess){
                std::printf("MemoryOwner::failed to allocate %zu bytes on device: %s (%s)\n", 
                BytesData, cudaGetErrorString(err), cudaGetErrorName(err));
                throw std::bad_alloc();
            }
        }
    };
    
    __host__
    ~MemoryOwner(){
        if constexpr(Location == MemoryOn::Host)
            free(this->memPtr);
        else
            cudaFree(this->memPtr);
    }

    MemoryOwner           (const MemoryOwner& ) = delete;
    MemoryOwner& operator=(const MemoryOwner& ) = delete;
    MemoryOwner                 (MemoryOwner&&) = delete;
    MemoryOwner& operator=      (MemoryOwner&&) = delete;    

    __host__
    void zeros(){
        if constexpr(Location == MemoryOn::Host)
            std::memset(this->memPtr, 0, BytesData);
        else{
            cudaError_t err = cudaMemset(memPtr, 0, BytesData);
            if (err != cudaSuccess) {
                std::fprintf(stderr, "MemoryOwner::zeros failed: %s on device\n", cudaGetErrorString(err));
            }
        }
    }

};

template<MemoryOn Location, typename View>
struct Owning: MemoryOwner<Location, View::bytesData>, View {
    using BaseOwner = MemoryOwner<Location, View::bytesData>;
    Owning(): BaseOwner(), View(static_cast<View::ElementType*>(BaseOwner::memPtr)){}
};

