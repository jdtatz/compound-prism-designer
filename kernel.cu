#define _USE_MATH_DEFINES
#include <math.h>
#include <stdint.h>

constexpr int warp_size = 32;
constexpr unsigned mask = 0xffffffff;

extern __shared__ float shared[];

__device__ float operator_add(const float a, const float b) {
    return a + b;
}

__device__ unsigned ilog2(unsigned v){
    return 31 - __clz(v);
}

__device__ bool is_pow_2(unsigned v){
    return !(v & (v - 1));
}

__device__ float reduce(float (*func)(const float, const float), float val, const unsigned width){
    const unsigned laneid = threadIdx.x & 0x1f;
    if(width <= warp_size && is_pow_2(width)){
        for(int i=0; i < ilog2(width); i++){
            val = func(val, __shfl_xor_sync(mask, val, 1 << i));
        }
        return val;
    } else if (width <= warp_size){
        unsigned closest_pow2 = 1 << ilog2(width);
        unsigned diff = width - closest_pow2;
        auto temp = __shfl_down_sync(mask, val, closest_pow2);
        if (laneid < diff){
            val = func(val, temp);
        }
        for(int i=0; i < ilog2(width); i++){
            val = func(val, __shfl_xor_sync(mask, val, 1 << i));
        }
        return __shfl_sync(mask, val, 0);
    } else {
        unsigned last_warp_size = width % warp_size;
        unsigned warp_count = width / warp_size + (last_warp_size ? 1 : 0);
        __syncthreads();
        unsigned tid = threadIdx.x + threadIdx.y * blockDim.x;
        __syncthreads();
        if (last_warp_size == 0 || tid < width - last_warp_size) {
            for(int i=0; i < ilog2(warp_size); i++)
                val = func(val, __shfl_xor_sync(mask, val, 1 << i));
        } else if (is_pow_2(last_warp_size)) {
            for(int i=0; i < ilog2(last_warp_size); i++)
                val = func(val, __shfl_xor_sync(mask, val, 1 << i));
        } else{
            int closest_lpow2 = 1 << ilog2(last_warp_size);
            auto temp = __shfl_down_sync(mask, val, closest_lpow2);
            if (laneid < last_warp_size - closest_lpow2)
                val = func(val, temp);
            for(int i=0; i < ilog2(closest_lpow2); i++)
                val = func(val, __shfl_xor_sync(mask, val, 1 << i));
        }
        if (laneid == 0)
            shared[tid / warp_size] = val;
        __syncthreads();
        val = shared[0];
        for(int i=1; i < warp_count; i++)
            val = func(val, shared[i]);
        return val;
    }
}


__device__ void init_xoroshiro128p_state(uint64_t *rng, uint64_t seed) {
    uint64_t z = seed + static_cast<uint64_t>(0x9E3779B97F4A7C15);
    z = (z ^ (z >> static_cast<uint32_t>(30))) * static_cast<uint64_t>(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> static_cast<uint32_t>(27))) * static_cast<uint64_t>(0x94D049BB133111EB);
    z = z ^ (z >> static_cast<uint32_t>(31));

    rng[0] = z;
    rng[1] = z;
}

__device__ uint64_t rotl(uint64_t x, uint32_t k) {
    return (x << k) | (x >> static_cast<uint32_t>(64 - k));
}

__device__ uint64_t xoroshiro128p_next(uint64_t *rng) {
    uint64_t s0 = rng[0];
    uint64_t s1 = rng[1];
    uint64_t result = s0 + s1;

    s1 ^= s0;
    rng[0] = rotl(s0, static_cast<uint32_t>(55)) ^ s1 ^ (s1 << static_cast<uint32_t>(14));
    rng[1] = rotl(s1, static_cast<uint32_t>(36));

    return result;
}

__device__ void xoroshiro128p_jump(uint64_t *rng) {
    constexpr uint64_t XOROSHIRO128P_JUMP[] = {0xbeac0467eba5facb, 0xd86b048b86aa9922};
    uint64_t s0 = 0;
    uint64_t s1 = 0;
    for(int i=0; i < 2; i++){
        for(int b=0; b < 64; b++){
            if(XOROSHIRO128P_JUMP[i] & (static_cast<uint64_t>(1) << static_cast<uint32_t>(b))){
                s0 ^= rng[0];
                s1 ^= rng[1];
            }
            xoroshiro128p_next(rng);
        }
    }
    rng[0] = s0;
    rng[1] = s1;
}

__device__ float xoroshiro128p_uniform_float32(uint64_t *rng){
    uint64_t x = xoroshiro128p_next(rng);
    double y = (x >> static_cast<uint32_t>(11)) * (static_cast<double>(1) / (static_cast<uint64_t>(1) << static_cast<uint32_t>(53)));
    return static_cast<float>(y);
}

__device__ float xoroshiro128p_normal_float32(uint64_t *rng){
    float u1 = xoroshiro128p_uniform_float32(rng);
    float u2 = xoroshiro128p_uniform_float32(rng);
    float z0 = sqrtf(-2 * logf(u1)) * cosf(2 * M_PI * u2);
    return z0;
}


struct V2 {
    float x, y;

    V2(): x(0), y(0) {}

    V2(float x, float y): x(x), y(y) {}

    V2 operator+(const V2 &rhs) const {
        return V2(this->x + rhs.x, this->y + rhs.y);
    }

    V2 operator-(const V2 &rhs) const {
        return V2(this->x - rhs.x, this->y - rhs.y);
    }

    V2 operator*(const V2 &rhs) const {
        return V2(this->x * rhs.x, this->y * rhs.y);
    }

    V2 operator*(const float rhs) const {
        return V2(this->x * rhs, this->y * rhs);
    }

    V2 operator/(const float rhs) const {
        return V2(this->x / rhs, this->y / rhs);
    }

    float dot(const V2 &rhs) const {
        return this->x * rhs.x + this->y * rhs.y;
    }

    float square() const {
        return this->x * this->x + this->y * this->y;
    }

    float length() const {
        return sqrtf(this->square());
    }
};

struct Ray {
    V2 p, v;
    float T;

    Ray(): p(), v(), T(0) {}

    Ray(V2 p, V2 v, float T): p(p), v(v), T(T) {}

    Ray intersect_surface(const V2 vertex, const V2 normal, float n1, float n2) const {
        float r = n1 / n2;
        float ci = -this->v.dot(normal);
        float d = (this->p - vertex).dot(normal) / ci;
        V2 p = this->p + this->v * d;
        float cr = sqrtf(1 - r * r * (1 - ci * ci));
        V2 v = this->v * r + normal * (r * ci - cr);
        float fresnel_rs = ((n1 * ci - n2 * cr) / (n1 * ci + n2 * cr));
        float fresnel_rp = ((n1 * cr - n2 * ci) / (n1 * cr + n2 * ci));
        float transmittance = 1 - (fresnel_rs * fresnel_rs + fresnel_rp * fresnel_rp) / 2;
        return Ray(p, v, this->T * transmittance);
    }

    Ray intersect_lens(const V2 midpt, const V2 normal, float curvature, float n1, float n2) const {
        float r = n1 / n2;
        float diameter = 1 / fabsf(normal.x);
        float lens_radius = diameter / (2 * curvature);
        float rs = sqrtf(lens_radius * lens_radius - diameter * diameter / 4);
        V2 center = midpt + normal * rs;
        V2 delta = this->p - center;
        float ud = this->v.dot(delta);
        float d = -ud + sqrtf(ud * ud - delta.square() + lens_radius * lens_radius);
        V2 p = this->p + this->v * d;
        V2 snorm = (center - this->p) / lens_radius;
        float ci = -this->v.dot(snorm);
        float cr = sqrtf(1 - r * r * (1 - ci * ci));
        V2 v = this->v * r + normal * (r * ci - cr);
        float fresnel_rs = ((n1 * ci - n2 * cr) / (n1 * ci + n2 * cr));
        float fresnel_rp = ((n1 * cr - n2 * ci) / (n1 * cr + n2 * ci));
        float transmittance = 1 - (fresnel_rs * fresnel_rs + fresnel_rp * fresnel_rp) / 2;
        return Ray(p, v, this->T * transmittance);
    }

    float intersect_spectrometer(const Ray upper_ray, const Ray lower_ray, float spec_angle, float spec_length) const {
        V2 normal(-cosf(spec_angle), -sinf(spec_angle));
        float det = upper_ray.v.y * lower_ray.v.x - upper_ray.v.x * lower_ray.v.y;
        float d2 = (upper_ray.v.x * (lower_ray.p.y - upper_ray.p.y - spec_length * normal.x) +
                    upper_ray.v.y * (upper_ray.p.x - lower_ray.p.x - spec_length * normal.y)) / det;
        V2 l_vertex = lower_ray.p + lower_ray.v * d2;
        float ci = -this->v.dot(normal);
        float d = (this->p - l_vertex).dot(normal) / ci;
        V2 p = this->p + this->v * d;
        V2 vdiff = p - l_vertex;
        float spec_pos = sqrtf(vdiff.dot(vdiff)) / spec_length;
        return spec_pos;
    }
};

struct Config {
    float start, radius, theta0, sheight, lower_bound, upper_bound;
    float weight_deviation, weight_linearity, weight_transmittance, weight_spot;
    float dr, factor;
    int steps;
    Config(const float*config) : start(config[0]), radius(config[1]), theta0(config[2]), sheight(config[3]),
    lower_bound(config[4]), upper_bound(config[5]), weight_deviation(config[6]), weight_linearity(config[7]),
    weight_transmittance(config[8]), weight_spot(config[9]), dr(config[10]), factor(config[11]),
    steps(static_cast<int>(config[12])) {}
};

__device__ float nonlinearity(){
    const unsigned nwaves = blockDim.x;
    const unsigned tix = threadIdx.x;
    float val = shared[tix];
    float err;
    if (tix == 0)
        err = (2 * shared[2] + 2 * val - 4 * shared[1]);
    else if (tix == nwaves - 1)
        err = (2 * val - 4 * shared[nwaves - 2] + 2 * shared[nwaves - 3]);
    else if (tix == 1)
        err = (shared[3] - 3 * val + 2 * shared[0]);
    else if (tix == nwaves - 2)
        err = (2 * shared[nwaves - 1] - 3 * val + shared[nwaves - 4]);
    else
        err = (shared[tix + 2] + shared[tix - 2] - 2 * val);
    return sqrtf(reduce(operator_add, err * err, nwaves)) / 4;
}


template <int prism_count>
__device__ float merit_error(const Config &config, const float *n, const float *params){
    __syncthreads();
    const unsigned tix = threadIdx.x;
    const unsigned tiy = threadIdx.y;
    const unsigned nwaves = blockDim.x;
    __shared__ Ray shared_rays[3];
    // Initial Surface
    float n1 = 1, n2 = n[0];
    V2 normal(-cosf(params[2]), -sinf(params[2]));
    const float start = (tiy == 0 ? config.start : (tiy == 1 ? (config.start + config.radius) : (config.start - config.radius)));
    V2 vertex(fabsf(normal.y / normal.x), 1);
    Ray inital({0, start}, {cosf(config.theta0), sinf(config.theta0)}, 1);
    Ray ray = inital.intersect_surface(vertex, normal, n1, n2);
    if (__syncthreads_or(ray.p.y <= 0 || 1 <= ray.p.y || isnan(ray.T)))
        return INFINITY;
    // Inner Surfaces
    for(int i=1; i < prism_count; i++) {
        n1 = n2;
        n2 = n[i];
        normal.x = -cosf(params[2 + i]);
        normal.y = -sinf(params[2 + i]);
        vertex.x += fabsf(normal.y / normal.x);
        vertex.y = (i + 1) % 2;
        ray = ray.intersect_surface(vertex, normal, n1, n2);
        if (__syncthreads_or(ray.p.y <= 0 || 1 <= ray.p.y || isnan(ray.T)))
            return INFINITY;
    }
    // Last / Convex Surface
    n1 = n2;
    n2 = 1;
    normal.x = -cosf(params[2 + prism_count]);
    normal.y = -sinf(params[2 + prism_count]);
    float diff = fabsf(normal.y / normal.x);
    vertex.x += diff;
    vertex.y = (prism_count + 1) % 2;
    V2 midpt(vertex.x - diff / 2, 0.5);
    float curvature = params[0];
    ray = ray.intersect_lens(midpt, normal, curvature, n1, n2);
    float diameter = 1 / fabsf(normal.x);
    bool on_lens = (ray.p - midpt).square() <= (diameter * diameter / 4);
    if (__syncthreads_or(!on_lens || isnan(ray.T)))
        return INFINITY;
    // Spectrometer
    if (tix == nwaves / 2 && tiy == 0)
        shared_rays[0] = ray;
    else if( tix == 0 && tiy == 0)
        shared_rays[1] = ray;
    else if(tix == nwaves - 1 && tiy == 0)
        shared_rays[2] = ray;
    __syncthreads();
    bool keep = shared_rays[1].p.y > shared_rays[2].p.y;
    Ray upper_ray = keep ? shared_rays[1] : shared_rays[2];
    Ray lower_ray = keep ? shared_rays[2] : shared_rays[1];
    float n_spec_pos = ray.intersect_spectrometer(upper_ray, lower_ray, params[1], config.sheight);
    shared[tiy * nwaves + tix] = n_spec_pos;
    __syncthreads();
    float spot_size = fabsf(shared[nwaves + tix] - shared[2 * nwaves + tix]);
    float nonlin = nonlinearity();
    float deviation = fabsf(shared_rays[0].v.y);
    float mean_transmittance = reduce(operator_add, ray.T, nwaves) / nwaves;
    float mean_spot_size = reduce(operator_add, spot_size, nwaves) / nwaves;
    __syncthreads();
    return config.weight_deviation * deviation + config.weight_linearity * nonlin + config.weight_transmittance * (1 - mean_transmittance) + config.weight_spot * mean_spot_size;
}

template <int prism_count>
__device__ V2 random_search(const Config &config, const float *n, uint64_t *rng, const float lbound, const float ubound) {
    constexpr int param_count = prism_count + 2;
    const unsigned tid = threadIdx.x + threadIdx.y * blockDim.x;
    const unsigned rid = 2 * (blockIdx.x * param_count + tid);
    float best = 0;
    __shared__ float trial[param_count];
    if(tid < param_count) {
        float rand = xoroshiro128p_uniform_float32(rng + rid);
        best = lbound + rand * (ubound - lbound);
        trial[tid] = best;
    }
    __syncthreads();
    float bestVal = merit_error<prism_count>(config, n, trial);
    for(int rs=0; rs < config.steps; rs++) {
        if(tid < param_count){
            float xi = xoroshiro128p_normal_float32(rng + rid);
            float sphere = xi / sqrtf(reduce(operator_add, xi * xi, param_count));
            float test = best + sphere * config.dr * expf(-static_cast<float>(rs) * config.factor);
            trial[tid] = max(min(test, ubound), lbound);
        }
        __syncthreads();
        float trialVal = merit_error<prism_count>(config, n, trial);
        if(tid < param_count && trialVal < bestVal) {
            bestVal = trialVal;
            best = trial[tid];
        }
    }
    return {best, bestVal};
}

template <int prism_count>
__device__ void optimize(const float *configuration, const float *ns, const size_t nglass, float *out, size_t start, size_t stop, uint64_t rng_seed){
    constexpr int param_count = prism_count + 2;
    const unsigned nwaves = blockDim.x;
    const unsigned tid = threadIdx.x + threadIdx.y * blockDim.x;
    const unsigned tix = threadIdx.x;
    const unsigned bid = blockIdx.x;
    const unsigned bcount = gridDim.x;
    const Config config(configuration);
    const float lbound = (tid == 0 ? 0 : (tid == 1 ? -M_PI_2 : (tid % 2 == 1 ? config.lower_bound : -config.upper_bound)));
    const float ubound = (tid == 0 ? 1 : (tid == 1 ? M_PI_2 : (tid % 2 == 1 ? config.upper_bound : -config.lower_bound)));
    float bestVal = INFINITY;
    float n[param_count];
    uint64_t rng[2];
    if(tid < param_count){
        init_xoroshiro128p_state(rng, rng_seed);
        for(unsigned i=0; i < tid; i++) {
            xoroshiro128p_jump(rng);
        }
    }
    for(int index=start+bid; index < stop; index += bcount) {
        size_t tot = 1;
        for (int i = 0; i < prism_count; i++) {
            n[i] = ns[((index / tot) % nglass) * nwaves + tix];
            tot *= nglass;
        }
        V2 opt = random_search<prism_count>(config, n, rng, lbound, ubound);
        float xmin = opt.x, fx = opt.y;
        if (tid < param_count && fx < bestVal){
            bestVal = fx;
            const unsigned oid = bid*(param_count + 2);
            if(tid == 0){
                out[oid] = fx;
                out[oid + 1] = index;
                out[oid + 2] = xmin;
            } else {
                out[oid + 1 + tid] = xmin;
            }
        }
        __syncthreads();
    }
}

extern "C" __global__ void call_optimize(const int prism_count, const float *configuration, const float *ns, const size_t nglass, float *out, size_t start, size_t stop, uint64_t rng_seed) {
    if(prism_count == 1)
        optimize<1>(configuration, ns, nglass, out, start, stop, rng_seed);
    else if(prism_count == 2)
        optimize<2>(configuration, ns, nglass, out, start, stop, rng_seed);
    else if(prism_count == 3)
        optimize<3>(configuration, ns, nglass, out, start, stop, rng_seed);
    else if(prism_count == 4)
        optimize<4>(configuration, ns, nglass, out, start, stop, rng_seed);
    else if(prism_count == 5)
        optimize<5>(configuration, ns, nglass, out, start, stop, rng_seed);
}
