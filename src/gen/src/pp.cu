#include <Eigen/Eigen>
#include <cuda_runtime.h>
#include <stdio.h>
#include <ctime>
#include <unistd.h>

#define THREAD_NUM 256

using namespace Eigen;
using namespace std;

void __checkCudaError ( cudaError_t result_t, const char * file, const int line )
{
    std::string error_string;

    if ( cudaSuccess != result_t && cudaErrorCudartUnloading != result_t )
    {
        fprintf ( stderr, "\x1B[31m CUDA error encountered in file '%s', line %d\n Error %d: %s\n Terminating FIRE!\n \x1B[0m", file, line, result_t,
               cudaGetErrorString ( result_t ) );
        printf("CUDA error encountered: %s", cudaGetErrorString ( result_t ) );
        printf(". Terminating application.\n");
        throw std::runtime_error ( "checkCUDAError : ERROR: CUDA Error" );
    }
}

__constant__ int   size_pos[2];
__constant__ int   xy_num;
__constant__ int   start_idx[2];
__constant__ float position[2];
__constant__ float length_pos[2];
__constant__ float resolution_pos;
__constant__ float resolution_pos_inv;
__constant__ float origin[2];
__constant__ float normal_radius;
__constant__ float max_curvature;
__constant__ float min_cosxi;

__device__ int signFunc(const int val)
{
    return static_cast<int>(0 < val) - static_cast<int>(val < 0);
} 

__host__ __device__ void wrapIndexOne(int& idx, int buffer_size)
{
    if (idx < buffer_size)
    {
        if(idx >= 0)
            return;
        else if(idx >= -buffer_size)
        {
            idx += buffer_size;
            return;
        }
        else
        {
            idx = idx % buffer_size;
            idx += buffer_size;
        }
    }
    else if(idx < buffer_size*2)
    {
        idx -= buffer_size;
        return;
    }
    else
        idx = idx % buffer_size;
}

__device__ void boundPos(Vector2f& pos, const Vector2f& position)
{
    Vector2f positionShifted = pos - position;
    positionShifted[0] += origin[0];
    positionShifted[1] += origin[1];

    for (int i = 0; i < positionShifted.size(); i++)
    {
        float epsilon = 10.0 * 1e-10;
        if (fabs(pos(i)) > 1.0)
            epsilon *= fabs(pos(i));

        if (positionShifted(i) <= 0)
        {
            positionShifted(i) = epsilon;
            continue;
        }
        if (positionShifted(i) >= length_pos[i])
        {
            positionShifted(i) = length_pos[i] - epsilon;
            continue;
        }
    }

    pos = positionShifted + position;
    pos[0] -= origin[0];
    pos[1] -= origin[1];
}

__device__ void computerEigenvalue(float *pMatrix, int nDim, float *maxvector, float *curvature, float dbEps, int nJt)
{
    float pdblVects[9];
    float pdbEigenValues[3];
    
	for(int i = 0; i < nDim; i ++) 
	{   
		pdblVects[i*nDim+i] = 1.0f; 
		for(int j = 0; j < nDim; j ++) 
		{ 
			if(i != j)   
				pdblVects[i*nDim+j]=0.0f; 
		} 
	} 
 
	int nCount = 0;
	while(1)
	{
		float dbMax = pMatrix[1];
		int nRow = 0;
		int nCol = 1;
		for (int i = 0; i < nDim; i ++)
		{
			for (int j = 0; j < nDim; j ++)
			{
				float d = fabs(pMatrix[i*nDim+j]); 
 
				if((i!=j) && (d> dbMax)) 
				{ 
					dbMax = d;   
					nRow = i;   
					nCol = j; 
				} 
			}
		}
 
		if(dbMax < dbEps) 
			break;  
 
		if(nCount > nJt)
			break;
 
		nCount++;
 
		float dbApp = pMatrix[nRow*nDim+nRow];
		float dbApq = pMatrix[nRow*nDim+nCol];
		float dbAqq = pMatrix[nCol*nDim+nCol];
 
		float dbAngle = 0.5*atan2(-2*dbApq,dbAqq-dbApp);
		float dbSinTheta = sin(dbAngle);
		float dbCosTheta = cos(dbAngle);
		float dbSin2Theta = sin(2*dbAngle);
		float dbCos2Theta = cos(2*dbAngle);
 
		pMatrix[nRow*nDim+nRow] = dbApp*dbCosTheta*dbCosTheta + 
			dbAqq*dbSinTheta*dbSinTheta + 2*dbApq*dbCosTheta*dbSinTheta;
		pMatrix[nCol*nDim+nCol] = dbApp*dbSinTheta*dbSinTheta + 
			dbAqq*dbCosTheta*dbCosTheta - 2*dbApq*dbCosTheta*dbSinTheta;
		pMatrix[nRow*nDim+nCol] = 0.5*(dbAqq-dbApp)*dbSin2Theta + dbApq*dbCos2Theta;
		pMatrix[nCol*nDim+nRow] = pMatrix[nRow*nDim+nCol];
 
		for(int i = 0; i < nDim; i ++) 
		{ 
			if((i!=nCol) && (i!=nRow)) 
			{ 
				int u = i*nDim + nRow;	//p  
				int w = i*nDim + nCol;	//q
				dbMax = pMatrix[u]; 
				pMatrix[u]= pMatrix[w]*dbSinTheta + dbMax*dbCosTheta; 
				pMatrix[w]= pMatrix[w]*dbCosTheta - dbMax*dbSinTheta; 
			} 
		} 
 
		for (int j = 0; j < nDim; j ++)
		{
			if((j!=nCol) && (j!=nRow)) 
			{ 
				int u = nRow*nDim + j;	//p
				int w = nCol*nDim + j;	//q
				dbMax = pMatrix[u]; 
				pMatrix[u]= pMatrix[w]*dbSinTheta + dbMax*dbCosTheta; 
				pMatrix[w]= pMatrix[w]*dbCosTheta - dbMax*dbSinTheta; 
			} 
		}
 
		for(int i = 0; i < nDim; i ++) 
		{ 
			int u = i*nDim + nRow;		//p   
			int w = i*nDim + nCol;		//q
			dbMax = pdblVects[u]; 
			pdblVects[u] = pdblVects[w]*dbSinTheta + dbMax*dbCosTheta; 
			pdblVects[w] = pdblVects[w]*dbCosTheta - dbMax*dbSinTheta; 
		} 
 
	}
    
    int min_id = 0;
	float minEigenvalue;
    float sumEigenvalue = 0.0;

	for(int i = 0; i < nDim; i ++) 
	{   
		pdbEigenValues[i] = pMatrix[i*nDim+i];
        sumEigenvalue += pdbEigenValues[i];
        if(i == 0)
            minEigenvalue = pdbEigenValues[i];
        else
        {
            if(minEigenvalue > pdbEigenValues[i])
            {
                minEigenvalue = pdbEigenValues[i];
                min_id = i;	
            }
        }
    } 

    for(int i = 0; i < nDim; i ++) 
    {  
        maxvector[i] = pdblVects[min_id + nDim * i];
    }

    *curvature = 3.0 * minEigenvalue / sumEigenvalue;
}

__global__ void compute_map(int point_num, 
                            float* inpainted,
                            float* smooth,
                            float* ground)
{
    int map_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (map_idx < xy_num)
    {
        int map_idx_true = map_idx / size_pos[1] + (map_idx % size_pos[1]) * size_pos[0];
        Vector2i idx(map_idx/size_pos[1], map_idx%size_pos[1]);
        Vector2f origin_off(origin[0], origin[1]);
        origin_off[0] -= 0.5 * resolution_pos;
        origin_off[1] -= 0.5 * resolution_pos;
        if ( start_idx[0] != 0 || start_idx[1] != 0)
        {
            idx[0] -= start_idx[0];
            idx[1] -= start_idx[1];
            for (int i=0; i<2; i++)
                wrapIndexOne(idx[i], size_pos[i]);
        }
        Map<Vector2f> position_(position);
        Vector2f center = position_ + origin_off - (idx.cast<float>() * resolution_pos).matrix();
        Vector2f top_left = center.array() + normal_radius;
        Vector2f bottom_right = center.array() - normal_radius;
        boundPos(top_left, position_);
        boundPos(bottom_right, position_);
        Array2i sub_start;
        sub_start[0] = (int) ( (origin[0] + position[0] - top_left[0]) * resolution_pos_inv );
        sub_start[1] = (int) ( (origin[1] + position[1] - top_left[1]) * resolution_pos_inv );
        Array2i sub_end;
        sub_end[0] = (int) ( (origin[0] + position[0] - bottom_right[0]) * resolution_pos_inv );
        sub_end[1] = (int) ( (origin[1] + position[1] - bottom_right[1]) * resolution_pos_inv );
        if ( start_idx[0] != 0 || start_idx[1] != 0)
        {
            sub_start[0] += start_idx[0];
            sub_start[1] += start_idx[1];
            for (int i=0; i<2; i++)
                wrapIndexOne(sub_start[i], size_pos[i]);
            sub_end[0] += start_idx[0];
            sub_end[1] += start_idx[1];
            for (int i=0; i<2; i++)
                wrapIndexOne(sub_end[i], size_pos[i]);
            
            sub_start[0] -= start_idx[0];
            sub_start[1] -= start_idx[1];
            for (int i=0; i<2; i++)
                wrapIndexOne(sub_start[i], size_pos[i]);
            sub_end[0] -= start_idx[0];
            sub_end[1] -= start_idx[1];
            for (int i=0; i<2; i++)
                wrapIndexOne(sub_end[i], size_pos[i]);
        }
        Array2i buffer_size = sub_end - sub_start + Array2i::Ones();
        Vector3f mean_p = Vector3f::Zero();
        int cnt_p = 0;
        Vector3f temp_poses[800];
        for (int i=0; i<buffer_size[0]; i++)
        {
            for (int j=0; j<buffer_size[1]; j++)
            {
                Array2i tempIndex = sub_start + Array2i(i, j);
                if ( start_idx[0] != 0 || start_idx[1] != 0)
                {
                    Map<Array2i> start_index(start_idx);
                    tempIndex += start_index;
                    for (int i = 0; i < tempIndex.size(); i++)
                        wrapIndexOne(tempIndex(i), size_pos[i]);
                }
                
                // col-major
                int true_idx = tempIndex[1] * size_pos[0] + tempIndex[0];
                if ( start_idx[0] != 0 || start_idx[1] != 0)
                {
                    tempIndex[0] -= start_idx[0];
                    tempIndex[1] -= start_idx[1];
                    for (int i=0; i<2; i++)
                        wrapIndexOne(tempIndex[i], size_pos[i]);
                }
                Vector2f temp_pos = position_ + origin_off - (tempIndex.cast<float>() * resolution_pos).matrix();
                if ((temp_pos - center).norm() < normal_radius)
                {
                    Vector3f temp_pos3;
                    temp_pos3.head(2) = temp_pos;
                    temp_pos3[2] = inpainted[true_idx];
                    mean_p[0] += temp_pos[0];
                    mean_p[1] += temp_pos[1];
                    mean_p[2] += temp_pos3[2];
                    temp_poses[cnt_p] = temp_pos3;
                    cnt_p ++;
                }
            }
        }

        mean_p[0] = mean_p[0] / (float)cnt_p;
        mean_p[1] = mean_p[1] / (float)cnt_p;
        mean_p[2] = mean_p[2] / (float)cnt_p;
        
        smooth[map_idx_true] = mean_p[2];
        float dist2pos = (position_ - center).norm();
        if (cnt_p > 7)
        {
            float pMatrix[9] = {0};
            for(int i = 0; i < cnt_p; i ++)
            {
                pMatrix[0] = pMatrix[0] + (temp_poses[i][0] - mean_p[0]) * (temp_poses[i][0] - mean_p[0]);
                pMatrix[4] = pMatrix[4] + (temp_poses[i][1] - mean_p[1]) * (temp_poses[i][1] - mean_p[1]);
                pMatrix[8] = pMatrix[8] + (temp_poses[i][2] - mean_p[2]) * (temp_poses[i][2] - mean_p[2]);
                pMatrix[1] = pMatrix[1] + (temp_poses[i][0] - mean_p[0]) * (temp_poses[i][1] - mean_p[1]);
                pMatrix[2] = pMatrix[2] + (temp_poses[i][0] - mean_p[0]) * (temp_poses[i][2] - mean_p[2]);
                pMatrix[5] = pMatrix[5] + (temp_poses[i][1] - mean_p[1]) * (temp_poses[i][2] - mean_p[2]);
                pMatrix[3] = pMatrix[1];
                pMatrix[6] = pMatrix[2];
                pMatrix[7] = pMatrix[5];
            }
            
            float dbEps = 0.01;
            int nJt = 30;
            float normal_vec[3];
            float curvature;
            computerEigenvalue(pMatrix, 3, normal_vec, &curvature, dbEps, nJt);
            
            float cos_xi;
            if (normal_vec[2] > 0)
                cos_xi = normal_vec[2];
            else
                cos_xi = -normal_vec[2];

            if (cos_xi > min_cosxi && curvature < max_curvature)
            {
                ground[map_idx_true] = inpainted[map_idx_true];
            }
        }
    }

    return;
}

__global__ void init_map(float* elevation,
                         bool* map_nan,
                         float* inpainted,
                         float* smooth,
                         float* ground)
{
    int map_idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (map_idx < xy_num)
    {
        map_nan[map_idx] = true;
        elevation[map_idx] = NAN;
        inpainted[map_idx] = NAN;
        smooth[map_idx] = NAN;
        ground[map_idx] = NAN;
    }

    //! xulong
    if (map_idx == 0)
    {
        printf("device params:\n");
        printf("size_pos[0] = %d\n", size_pos[0]);
        printf("size_pos[1] = %d\n", size_pos[1]);
        printf("xy_num = %d\n", xy_num);
        printf("resolution_pos_inv = %f\n", resolution_pos_inv);
        printf("resolution_pos = %f\n", resolution_pos);
        printf("length_pos[0] = %f\n", length_pos[0]);
        printf("length_pos[1] = %f\n", length_pos[1]);
        printf("origin[0] = %f\n", origin[0]);
        printf("origin[1] = %f\n", origin[1]);
        printf("normal_radius = %f\n", normal_radius);
        printf("max_curvature = %f\n", max_curvature);
        printf("min_cosxi = %f\n", min_cosxi);
    }
}

__global__ void compute_elevation(int point_num, 
                                 float* points,
                                 float* elevation,
                                 bool* map_nan)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < point_num)
    {
        Vector3f point_world(points[idx*3], points[idx*3+1], points[idx*3+2]);
        Map<Vector2f> origin_(origin);
        Map<Array2i> start_index(start_idx);
        Map<Vector2f> positionv(position);
        
        Vector2f pos_map = origin_ + positionv - point_world.head(2);
        if (pos_map.x() >= 0.0 && pos_map.y() >= 0.0 && \
            pos_map.x() < length_pos[0] && pos_map.y() < length_pos[1])
        {
            // add to temp
            Array2i index;
            Array2f indexf = pos_map.array() * resolution_pos_inv;
            index(0) = (int)indexf(0);
            index(1) = (int)indexf(1);
            if ( start_index[0] != 0 || start_index[1] != 0)
            {
                index[0] += start_index[0];
                index[1] += start_index[1];
                for (int i=0; i<2; i++)
                    wrapIndexOne(index[i], size_pos[i]);
            }
            // int true_idx = index[0] * size_pos[1] + index[1];
            // col-major
            int true_idx = index[1] * size_pos[0] + index[0];
            if (map_nan[true_idx] == true)
            {
                map_nan[true_idx] = false;
                elevation[true_idx] = point_world.z();
            }
            else
            {
                if (elevation[true_idx] < point_world.z())
                {
                    elevation[true_idx] = point_world.z();
                }
            }
        }

    }
}

__global__ void compute_inpaint_spiral(float* elevation,
                                        bool* map_nan,
                                        float* inpainted)
{
    int map_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (map_idx < xy_num)
    {
        int map_idx_true = map_idx / size_pos[1] + (map_idx % size_pos[1]) * size_pos[0];
        if (map_nan[map_idx_true])
        {
            inpainted[map_idx_true] = NAN;
            Array2i idx(map_idx/size_pos[1], map_idx%size_pos[1]);
            if ( start_idx[0] != 0 || start_idx[1] != 0)
            {
                idx[0] -= start_idx[0];
                idx[1] -= start_idx[1];
                for (int i=0; i<2; i++)
                    wrapIndexOne(idx[i], size_pos[i]);
            }

            float radius2 = length_pos[0] * length_pos[0] + length_pos[1] * length_pos[1];
            int n_rings = (int) (sqrt(radius2) / resolution_pos) + 1;
            int distance_ = 0;
            do
            {
                bool get_nearest = false;
                distance_++;
                Array2i point(distance_, 0);
                Array2i pointInMap;
                Array2i normal;
                Map<Array2i> start_index(start_idx);
                do
                {
                    pointInMap = point + idx;
                    if (pointInMap[0] >= 0 && pointInMap[1] >= 0 &&
                        pointInMap[0] < size_pos[0] && pointInMap[1] < size_pos[1])
                    {
                         pointInMap += start_index;
                        for (int i=0; i<2; i++)
                            wrapIndexOne(pointInMap[i], size_pos[i]);
                        // col-major
                        int true_idx = pointInMap[1] * size_pos[0] + pointInMap[0];
                        if (!map_nan[true_idx])
                        {
                            get_nearest = true;
                            inpainted[map_idx_true] = elevation[true_idx];
                            break;
                        }
                    }

                    normal[0] = -signFunc(point[1]);
                    normal[1] = signFunc(point[0]);
                    if (normal[0] !=0 && (int)(Vector2f(point[0] + normal[0], point[1]).norm()) == distance_)
                        point[0] += normal[0];
                    if (normal[0] !=0 && (int)(Vector2f(point[0], point[1] + normal[1]).norm()) == distance_)
                        point[1] += normal[1];
                    else
                    {
                        point += normal;
                    }
                } while (point[0]!=distance_ || point[1]!=0);
                if (get_nearest)
                    break;
            }while(distance_<n_rings);
        }
        else
        {
            inpainted[map_idx_true] = elevation[map_idx_true];
        }
    }
}

void gpuInit(const Vector2i& size_pos_,
            const Vector2f& length_pos_,
            float resolution_pos_,
            const Array2i& start_i,
            const Vector2f& position_,
            float normal_radius_,
            float max_curvature_,
            float min_cosxi_)
{
    // set up
    cudaSetDevice(0);
    // get params
    int xy_num_ = size_pos_[0] * size_pos_[1];
    float res_pos_inv_ = 1.0f / resolution_pos_;
    float origin_[2];
    origin_[0] = 0.5 * length_pos_[0];
    origin_[1] = 0.5 * length_pos_[1];
    cudaMemcpyToSymbol(size_pos, &size_pos_, sizeof(int) * 2);
    cudaMemcpyToSymbol(xy_num, &xy_num_, sizeof(int));
    cudaMemcpyToSymbol(resolution_pos_inv, &res_pos_inv_, sizeof(float));
    cudaMemcpyToSymbol(resolution_pos, &resolution_pos_, sizeof(float));
    cudaMemcpyToSymbol(length_pos, &length_pos_, sizeof(float) * 2);
    cudaMemcpyToSymbol(origin, &origin_, sizeof(float) * 2);
    cudaMemcpyToSymbol(start_idx, &start_i, sizeof(int) * 2);
    cudaMemcpyToSymbol(position, &position_, sizeof(float) * 2);
    cudaMemcpyToSymbol(normal_radius, &normal_radius_, sizeof(float));
    cudaMemcpyToSymbol(max_curvature, &max_curvature_, sizeof(float));
    cudaMemcpyToSymbol(min_cosxi, &min_cosxi_, sizeof(float));
    cudaDeviceSynchronize();

    return;
}

void processMap(int& point_num,
                float* points,
                MatrixXf& map,
                MatrixXf& inpainted,
                MatrixXf& smooth,
                MatrixXf& ground)
{
    int xy_num_ = map.rows() * map.cols();

    float* dev_points;
    float* dev_elevation;
    float* dev_inpainted;
    float* dev_smooth;
    float* dev_ground;
    bool* dev_nan;
    cudaMalloc((void**)&dev_points, point_num * sizeof(float) * 3);
    cudaMalloc((void**)&dev_elevation, xy_num_ * sizeof(float));
    cudaMalloc((void**)&dev_inpainted, xy_num_ * sizeof(float));
    cudaMalloc((void**)&dev_smooth, xy_num_ * sizeof(float));
    cudaMalloc((void**)&dev_ground, xy_num_ * sizeof(float));
    cudaMalloc((void**)&dev_nan, xy_num_ * sizeof(bool));
    cudaMemcpy(dev_points, points, point_num * sizeof(float) * 3, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    int blocksPerGrid =(xy_num_ + THREAD_NUM - 1) / THREAD_NUM;
	init_map<<<blocksPerGrid, THREAD_NUM>>>(dev_elevation, dev_nan, dev_inpainted, dev_smooth, dev_ground);
    cudaDeviceSynchronize();

    blocksPerGrid =(point_num + THREAD_NUM - 1) / THREAD_NUM;
	compute_elevation<<<blocksPerGrid, THREAD_NUM>>>(point_num, dev_points, dev_elevation, dev_nan);
    cudaDeviceSynchronize();

    blocksPerGrid = (xy_num_ + THREAD_NUM - 1) / THREAD_NUM;
    compute_inpaint_spiral<<<blocksPerGrid, THREAD_NUM>>>(dev_elevation, dev_nan, dev_inpainted);
    cudaDeviceSynchronize();
	compute_map<<<blocksPerGrid, THREAD_NUM>>>(xy_num_, dev_inpainted,  dev_smooth, dev_ground);
    cudaDeviceSynchronize();

    cudaMemcpy(map.data(), dev_elevation, xy_num_ * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(inpainted.data(), dev_inpainted, xy_num_ * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(smooth.data(), dev_smooth, xy_num_ * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(ground.data(), dev_ground, xy_num_ * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    cudaFree(dev_points);
    cudaFree(dev_elevation);
    cudaFree(dev_inpainted);
    cudaFree(dev_smooth);
    cudaFree(dev_ground);
    cudaFree(dev_nan);
    return;
}