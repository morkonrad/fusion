#include "f_wavelet.h"

void F_Wavelet::Initialize(const int &rows, const int &cols, const char *wave, const int &n_levels)
{
    wave_init(wave);

    wt_init_dummy(rows, cols, n_levels);

    checkCudaErrors(cudaStreamCreate(&stream1));
    checkCudaErrors(cudaStreamCreate(&stream2));

    //Dimensions of image at each level to allocate device memory at of mayimum respective sizes
    int cA_rows = wt_fus->length[wt_fus->n_levels][0];

    int Y = cA_rows*cols;

    checkCudaErrors(cudaMalloc((void **)&d_cH_v, Y * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_cH_ir, Y * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_cL_v, Y * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_cL_ir, Y * sizeof(float)));

    checkCudaErrors(cudaHostAlloc((void**)&fused, wt_fus->siglength * sizeof(float), cudaHostAllocDefault));

    ROI = {cols, rows};

    //Color
    Image_V_C = nppiMalloc_8u_C4(cols, rows, &step_C);
    Image_IR_C = nppiMalloc_8u_C4(cols, rows, &step_C);

    //Gray
    Image_V_Gr = nppiMalloc_8u_C1(cols, rows, &step_G);
    Image_IR_Gr = nppiMalloc_8u_C1(cols, rows, &step_G);

    //Floating point
    Image_V_F = nppiMalloc_32f_C1(cols, rows, &step_F);
    Image_IR_F = nppiMalloc_32f_C1(cols, rows, &step_F);

}

int wmaxiter(const int &sig_len, const int &filt_len) {
    int lev;
    double temp;

    temp = log((double)sig_len / ((double)filt_len - 1.0)) / log(2.0);
    lev = (int)temp;

    return lev;
}

int F_Wavelet::check_iter(const int &rows, const int &cols, const int &J)
{
    if (J > 100) {

        std::cout<<"Number of levels must not exceed 99"<<std::endl;
        exit(-1);
    }

    int f_size = wave_obj->filtlength;
    int MaxIter = std::max(wmaxiter(rows, f_size), wmaxiter(cols, f_size));

    if (J > MaxIter) {

        std::cout<<"\n Error - The Signal Can only be iterated %d times using this wavelet. Exiting\n"<< MaxIter<<std::endl; //TODO: MAke it QMessage box
        exit(-1);
    }
    return MaxIter;
}

void F_Wavelet::cu_filt_alloc() {

    //Creating constant device memory for filter
    int retval = wave_obj->filtlength;

    checkCudaErrors(cudaMalloc(&d_lpd, retval * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_hpd, retval * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_lpr, retval * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_hpr, retval * sizeof(float)));

    checkCudaErrors(cudaMemcpy(d_lpd, wave_obj->lpd, retval * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_hpd, wave_obj->hpd, retval * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_lpr, wave_obj->lpr, retval * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_hpr, wave_obj->hpr, retval * sizeof(float), cudaMemcpyHostToDevice));

    //Setting filters with constant memory
    set_filters(d_lpd, d_hpd, d_lpr, d_hpr);
}

void F_Wavelet::wave_init(const char* wname) {


    int filt_len = 0;

    if (wname != NULL) {
        filt_len = filtlength(wname);
    }

    wave_obj = (wave_object)malloc(sizeof(struct wave_set) + sizeof(float)* 4 * filt_len);

    wave_obj->filtlength = filt_len;

    strcpy(wave_obj->wname, wname);

    if (wname != NULL) {
        filtcoef(wname,wave_obj->params,wave_obj->params+ filt_len,wave_obj->params+2* filt_len,wave_obj->params+3* filt_len);
    }

    wave_obj->lpd = &wave_obj->params[0];
    wave_obj->hpd = &wave_obj->params[filt_len];
    wave_obj->lpr = &wave_obj->params[2 * filt_len];
    wave_obj->hpr = &wave_obj->params[3 * filt_len];

    this->cu_filt_alloc();

}

void F_Wavelet::wave_free(){

    free(wave_obj);

    cudaFree(d_lpd); cudaFree(d_hpd); cudaFree(d_lpr); cudaFree(d_hpr);

}

void F_Wavelet::wave_reinit(const char *wname){

    this->wave_free();

    this->wave_init(wname);

}

void F_Wavelet::set_n_lev(int n_levels){


    int i = n_levels;
    int r = wt_fus->length[n_levels + 1][0] = wt_fus->rows;
    int c = wt_fus->length[n_levels + 1][1] = wt_fus->cols;
    wt_fus->outlength = 0;

    wt_fus->MaxIter = check_iter(r, c, n_levels);

    while (i > 0) {

        r = (int)ceil((double)r / 2.0);
        wt_fus->length[i][0] = r;

        c = (int)ceil((double)c / 2.0);
        wt_fus->length[i][1] = c;

        wt_fus->outlength += 3 * r*c;
        i--;
    }

    wt_fus->length[0][0] = wt_fus->length[1][0];
    wt_fus->length[0][1] = wt_fus->length[1][1];
    wt_fus->outlength += wt_fus->length[0][0] * wt_fus->length[0][1];

    wt_fus->n_levels = n_levels;   //Num levels
    wt_fus->lenlength = n_levels + 2;

    _alloc_size_vis = wt_fus->outlength;
    _alloc_size_ir = wt_fus->outlength;

    checkCudaErrors(cudaMalloc((void **)&d_ip_v, wt_fus->outlength * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_ip_ir, wt_fus->outlength * sizeof(float)));

}

void F_Wavelet::dev_mem_free(){

    cudaFree(d_ip_v);
    cudaFree(d_ip_ir);
}

void F_Wavelet::wt_init_dummy(const int &rows,const int &cols, const int &J) {

    wt_fus = (wt_object)malloc(sizeof(struct wt_set));

    wt_fus->siglength = rows*cols;	//Input signal length//Input signal length

    wt_fus->rows = rows;
    wt_fus->cols = cols;

    this->set_n_lev(J);
}

void F_Wavelet::wt_free(){

    free(wt_fus);

    dev_mem_free();
    cudaFree(d_cL_v); cudaFree(d_cH_v); cudaFree(d_cL_ir); cudaFree(d_cH_ir);

    wave_free();

    nppiFree(Image_V_C);nppiFree(Image_IR_C);nppiFree(Image_IR_Gr);nppiFree(Image_V_Gr);
    nppiFree(Image_V_F);nppiFree(Image_IR_F);

    cudaStreamDestroy(stream1); cudaStreamDestroy(stream2);
}

void F_Wavelet::dwt() {

    int J, N, iter, rows, cols, cA_rows, cA_cols, app_len;

    int f_len = wave_obj->filtlength;
    int halo_steps_x = (f_len / (2 * X_BLOCKDIM_X)) + 1;
    if (f_len % (X_BLOCKDIM_X * 2) == 0) halo_steps_x--;

    int halo_steps_y = (f_len / (2 * Y_BLOCKDIM_Y)) + 1;
    if (f_len % (Y_BLOCKDIM_Y * 2) == 0) halo_steps_y--;

    J = wt_fus->n_levels;

    rows = wt_fus->length[J + 1][0], cols = wt_fus->length[J + 1][1];

    cA_rows = wt_fus->length[J][0], cA_cols = wt_fus->length[J][1];

    app_len = cA_rows*cA_cols; //Length of Approximate coefficients

    N = wt_fus->outlength;
    const auto out_length = wt_fus->outlength;
    for (iter = J; iter > 0; --iter) {

        N = N - 3 * app_len;   //To write blocks in right memory places

        DWT_Y_GPU(d_ip_v, rows, cols, cA_rows, wave_obj->filtlength, d_cL_v, d_cH_v, halo_steps_y, stream1);
        DWT_Y_GPU(d_ip_ir, rows, cols, cA_rows, wave_obj->filtlength, d_cL_ir, d_cH_ir, halo_steps_y, stream2);
        cudaDeviceSynchronize();

        DWT_X_GPU(d_cL_v, cA_rows, cols, cA_cols, wave_obj->filtlength, d_ip_v, d_ip_v + N, halo_steps_x, stream1);
        DWT_X_GPU(d_cH_v, cA_rows, cols, cA_cols, wave_obj->filtlength, d_ip_v + N + app_len, d_ip_v + N + 2 * app_len, halo_steps_x, stream1);
        cudaDeviceSynchronize();

        DWT_X_GPU(d_cL_ir, cA_rows, cols, cA_cols, wave_obj->filtlength, d_ip_ir, d_ip_ir + N, halo_steps_x, stream2);
        DWT_X_GPU(d_cH_ir, cA_rows, cols, cA_cols, wave_obj->filtlength, d_ip_ir + N + app_len, d_ip_ir + N + 2 * app_len, halo_steps_x, stream2);
        cudaDeviceSynchronize();        

        //dimensions at 'iter' level
        rows = wt_fus->length[iter][0], cols = wt_fus->length[iter][1];

        //Dimensions at next level
        cA_rows = wt_fus->length[iter - 1][0], cA_cols = wt_fus->length[iter - 1][1];

        app_len = cA_rows*cA_cols; //Length of Approximate coefficients
    }

    //Fuse
    DWT_compare(d_ip_v, d_ip_ir, app_len,out_length , stream1, stream2);
    cudaDeviceSynchronize();

}

void F_Wavelet::dwt2(std::vector<float> &ir_analysis, std::vector<float> &vis_analysis)
{
    int J, N, iter, rows, cols, cA_rows, cA_cols, app_len;

    int f_len = wave_obj->filtlength;
    int halo_steps_x = (f_len / (2 * X_BLOCKDIM_X)) + 1;
    if (f_len % (X_BLOCKDIM_X * 2) == 0) halo_steps_x--;

    int halo_steps_y = (f_len / (2 * Y_BLOCKDIM_Y)) + 1;
    if (f_len % (Y_BLOCKDIM_Y * 2) == 0) halo_steps_y--;

    J = wt_fus->n_levels;

    rows = wt_fus->length[J + 1][0], cols = wt_fus->length[J + 1][1];

    cA_rows = wt_fus->length[J][0], cA_cols = wt_fus->length[J][1];

    app_len = cA_rows*cA_cols; //Length of Approximate coefficients

    N = wt_fus->outlength;
    //const auto out_length = wt_fus->outlength;
    const auto out_length = app_len;
    for (iter = J; iter > 0; --iter) {

        N = N - 3 * app_len;   //To write blocks in right memory places

        DWT_Y_GPU(d_ip_v, rows, cols, cA_rows, wave_obj->filtlength, d_cL_v, d_cH_v, halo_steps_y, stream1);
        DWT_Y_GPU(d_ip_ir, rows, cols, cA_rows, wave_obj->filtlength, d_cL_ir, d_cH_ir, halo_steps_y, stream2);
        cudaDeviceSynchronize();

        DWT_X_GPU(d_cL_v, cA_rows, cols, cA_cols, wave_obj->filtlength, d_ip_v, d_ip_v + N, halo_steps_x, stream1);
        DWT_X_GPU(d_cH_v, cA_rows, cols, cA_cols, wave_obj->filtlength, d_ip_v + N + app_len, d_ip_v + N + 2 * app_len, halo_steps_x, stream1);
        cudaDeviceSynchronize();

        DWT_X_GPU(d_cL_ir, cA_rows, cols, cA_cols, wave_obj->filtlength, d_ip_ir, d_ip_ir + N, halo_steps_x, stream2);
        DWT_X_GPU(d_cH_ir, cA_rows, cols, cA_cols, wave_obj->filtlength, d_ip_ir + N + app_len, d_ip_ir + N + 2 * app_len, halo_steps_x, stream2);
        cudaDeviceSynchronize();

        //dimensions at 'iter' level
        rows = wt_fus->length[iter][0], cols = wt_fus->length[iter][1];

        //Dimensions at next level
        cA_rows = wt_fus->length[iter - 1][0], cA_cols = wt_fus->length[iter - 1][1];

        app_len = cA_rows*cA_cols; //Length of Approximate coefficients
    }


    vis_analysis.resize(_alloc_size_vis);
    checkCudaErrors(cudaMemcpy(vis_analysis.data(), d_ip_v, _alloc_size_vis * sizeof(float), cudaMemcpyDeviceToHost));

    ir_analysis.resize(_alloc_size_ir);
    checkCudaErrors(cudaMemcpy(ir_analysis.data(), d_ip_ir, _alloc_size_ir * sizeof(float), cudaMemcpyDeviceToHost));
}

void F_Wavelet::idwt() {

    int J, rows, cols, next_rows, next_cols, app_len, X;

    J = wt_fus->n_levels;
    int lf = wave_obj->filtlength;

    //dimensions of transformed coefficients at each level
    rows = wt_fus->length[1][0]; cols = wt_fus->length[1][1];
    next_rows = wt_fus->length[2][0]; next_cols = wt_fus->length[2][1];

    //Approximate coefficients length
    X = app_len = rows*cols;

    for (int iter = 0; iter < J; iter++)
    {
        IDWT_X_GPU_1(d_cL_v, d_ip_v, d_ip_v + X, rows, cols, next_cols, lf, stream1);
        IDWT_X_GPU_1(d_cH_v, d_ip_v + X + app_len, d_ip_v + X + 2 * app_len, rows, cols, next_cols, lf, stream2);
        cudaDeviceSynchronize();

        cudaDeviceSynchronize();

        IDWT_Y_GPU_1(d_ip_v, d_cL_v, d_cH_v, rows, next_cols, next_rows, lf, stream1);
        cudaDeviceSynchronize();

        X += 3 * app_len;
        app_len = next_rows * next_cols;
        rows = next_rows; cols = next_cols;

        next_rows = wt_fus->length[iter + 3][0]; next_cols = wt_fus->length[iter + 3][1];
    }

    checkCudaErrors(cudaMemcpyAsync(fused, d_ip_v, wt_fus->siglength * sizeof(float), cudaMemcpyDeviceToHost, stream1));
    //checkCudaErrors(cudaMemcpy(fused, d_ip_v, wt_fus->siglength * sizeof(float), cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();
}

float* F_Wavelet::Fuse_RGBA(const std::uint8_t *Img_v, const std::uint8_t *Img_ir,const uint& width, const uint& height)
{
    uchar4 *inp_v, *inp_ir;

    inp_v = (uchar4*)Img_v;
    inp_ir = (uchar4*)Img_ir;

    //Pinned memory
/*
    cudaHostRegister(inp_v, wt_fus->siglength * sizeof(uchar4), 0);
    cudaHostRegister(inp_ir, wt_fus->siglength * sizeof(uchar4), 0);
*/
    cudaMemcpy2DAsync(Image_V_C, step_C, inp_v, width * sizeof(uchar4), width * sizeof(uchar4), height, cudaMemcpyHostToDevice, stream1);
    cudaMemcpy2DAsync(Image_IR_C, step_C, inp_ir, width * sizeof(uchar4), width * sizeof(uchar4), height, cudaMemcpyHostToDevice, stream2);

    cudaDeviceSynchronize();

    nppiColorToGray_8u_C4C1R(Image_V_C, step_C, Image_V_Gr, step_G, ROI, coef);
    nppiColorToGray_8u_C4C1R(Image_IR_C, step_C, Image_IR_Gr, step_G, ROI, coef);

    cudaDeviceSynchronize();

    nppiConvert_8u32f_C1R(Image_V_Gr, step_G, Image_V_F, step_F, ROI);
    nppiConvert_8u32f_C1R(Image_IR_Gr, step_G, Image_IR_F, step_F, ROI);

    cudaDeviceSynchronize();

    cudaMemcpy2DAsync(d_ip_v, width * sizeof(float), (float*)Image_V_F, step_F, width * sizeof(float), height, cudaMemcpyDeviceToDevice, stream1);
    cudaMemcpy2DAsync(d_ip_ir, width * sizeof(float), (float*)Image_IR_F, step_F, width * sizeof(float), height, cudaMemcpyDeviceToDevice, stream2);

    cudaDeviceSynchronize();

    //DWT
    dwt();

    //IDWT
    idwt();

    return fused;

}

float* F_Wavelet::Fuse_Grayscale(const std::uint8_t *Img_v, const std::uint8_t *Img_ir,const uint& width, const uint& height)
{
    std::uint8_t *inp_v, *inp_ir;

    inp_v = (std::uint8_t*)Img_v;
    inp_ir = (std::uint8_t*)Img_ir;

    // Pinned memory
    /*
    auto err =cudaHostRegister(inp_v, wt_fus->siglength * sizeof(float), 0);
    if(err!=cudaSuccess){
        std::cout<<"Some error on cudaHostRegister: "<<err<<std::endl;
        std::exit(-1);
    }

    err = cudaHostRegister(inp_ir, wt_fus->siglength * sizeof(float), 0);
    if(err!=cudaSuccess){
        std::cout<<"Some error on cudaHostRegister: "<<err<<std::endl;
        std::exit(-1);
    }*/

    auto err = cudaMemcpy2DAsync(d_ip_v, width * sizeof(float), (void*)inp_v, width * sizeof(float), width* sizeof(float) , height, cudaMemcpyHostToDevice, stream1);
    //auto err = cudaMemcpy(d_ip_v, (void*)inp_v, width*height * sizeof(float), cudaMemcpyHostToDevice);
    if(err!=cudaSuccess){
        std::cout<<"Some error on copy to device DWT input img_visual: "<<err<<std::endl;
        std::exit(-1);
    }
    err = cudaMemcpy2DAsync(d_ip_ir, width * sizeof(float),(void*)inp_ir, width * sizeof(float), width* sizeof(float), height, cudaMemcpyHostToDevice, stream2);
    //err = cudaMemcpy(d_ip_ir, (void*)inp_ir, width*height * sizeof(float), cudaMemcpyHostToDevice);
    if(err!=cudaSuccess){
        std::cout<<"Some error on copy to device DWT input img_IR: "<<err<<std::endl;
        std::exit(-1);
    }

    cudaDeviceSynchronize();

    //DWT
    dwt();

    //IDWT
    idwt();

    return fused;
}
