#include "nonseparable.h"
#include "common.h"
#include <cstring>
#include <iostream>
//#ifdef SEPARATE_COMPILATION
// Required for separate compilation (see Makefile)
//#ifndef CONSTMEM_FILTERS_NS
//#define CONSTMEM_FILTERS_NS
//__constant__ DTYPE c_kern_LL[MAX_FILTER_WIDTH * MAX_FILTER_WIDTH];
//__constant__ DTYPE c_kern_LH[MAX_FILTER_WIDTH * MAX_FILTER_WIDTH];
//__constant__ DTYPE c_kern_HL[MAX_FILTER_WIDTH * MAX_FILTER_WIDTH];
//__constant__ DTYPE c_kern_HH[MAX_FILTER_WIDTH * MAX_FILTER_WIDTH];
//#endif
//#endif

// outer product of arrays "a", "b" of length "len"
DTYPE* w_outer(DTYPE* a, DTYPE* b, int len) {
    DTYPE* res = (DTYPE*) calloc(len*len, sizeof(DTYPE));
    for (int i = 0; i < len; i++) {
        for (int j = 0; j < len; j++) {
            res[i*len+j] = a[i]*b[j];
        }
    }
    return res;
}


/// Compute the four filters A, H, V, D  from a family name.
/// These filters are separable, i.e computed from 1D filters.
/// wname: name of the filter ("haar", "db3", "sym4", ...)
/// direction: 1 for forward transform, -1 for inverse transform
/// Returns : the filter width "hlen" if success ; a negative value otherwise.
int w_compute_filters(const char* wname, int direction, int do_swt) {
    if (direction == 0) {
        puts("ERROR: w_compute_filters(): please specify a direction for second argument : +1 for forward, -1 for inverse)");
        return -1;
    }
    int hlen = 0;
    DTYPE* f1_l; // 1D lowpass
    DTYPE* f1_h; // 1D highpass
    DTYPE* f2_a, *f2_h, *f2_v, *f2_d; // 2D filters

    // Haar filters has specific kernels
    if (!do_swt) {
        if ((!strcasecmp(wname, "haar")) || (!strcasecmp(wname, "db1")) || (!strcasecmp(wname, "bior1.1")) || (!strcasecmp(wname, "rbior1.1"))) {
            return 2;
        }
    }

    // Browse available filters (see filters.h)
    int i;
    for (i = 0; i < 72; i++) {
        if (!strcasecmp(wname, all_filters[i].wname)) {
            hlen = all_filters[i].hlen;
            if (direction > 0) {
                f1_l = all_filters[i].f_l;
                f1_h = all_filters[i].f_h;
            }
            else {
                f1_l = all_filters[i].i_l;
                f1_h = all_filters[i].i_h;
            }
            break;
        }
    }
    if (hlen == 0) {
        printf("ERROR: w_compute_filters(): unknown filter %s\n", wname);
        return -2;
    }

    // Create the separable 2D filters
    f2_a = w_outer(f1_l, f1_l, hlen);
    f2_h = w_outer(f1_l, f1_h, hlen); // CHECKME
    f2_v = w_outer(f1_h, f1_l, hlen);
    f2_d = w_outer(f1_h, f1_h, hlen);

    // Copy the filters to device constant memory
    std::memcpy(c_kern_LL, f2_a, hlen*hlen*sizeof(DTYPE));
    std::memcpy(c_kern_LH, f2_h, hlen*hlen*sizeof(DTYPE));
    std::memcpy(c_kern_HL, f2_v, hlen*hlen*sizeof(DTYPE));
    std::memcpy(c_kern_HH, f2_d, hlen*hlen*sizeof(DTYPE));

   /* 
    cudaMemcpyToSymbol(c_kern_LL, f2_a, hlen*hlen*sizeof(DTYPE), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(c_kern_LH, f2_h, hlen*hlen*sizeof(DTYPE), 0, cudaMemcpyHostToDevice); // CHECKME
    cudaMemcpyToSymbol(c_kern_HL, f2_v, hlen*hlen*sizeof(DTYPE), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(c_kern_HH, f2_d, hlen*hlen*sizeof(DTYPE), 0, cudaMemcpyHostToDevice);
*/
    return hlen;
}


int w_set_filters_forward_nonseparable(DTYPE* filter1, DTYPE* filter2, DTYPE* filter3, DTYPE* filter4, uint len) {

    std::memcpy(c_kern_LL, filter1, len*len*sizeof(DTYPE));
    std::memcpy(c_kern_LH, filter2, len*len*sizeof(DTYPE));
    std::memcpy(c_kern_HL, filter3, len*len*sizeof(DTYPE));
    std::memcpy(c_kern_HH, filter4, len*len*sizeof(DTYPE));
    
    /*
    if (cudaMemcpyToSymbol(c_kern_LL, filter1, len*len*sizeof(DTYPE), 0, cudaMemcpyHostToDevice) != cudaSuccess
            || cudaMemcpyToSymbol(c_kern_LH, filter2, len*len*sizeof(DTYPE), 0, cudaMemcpyHostToDevice) != cudaSuccess
            || cudaMemcpyToSymbol(c_kern_HL, filter3, len*len*sizeof(DTYPE), 0, cudaMemcpyHostToDevice) != cudaSuccess
            || cudaMemcpyToSymbol(c_kern_HH, filter4, len*len*sizeof(DTYPE), 0, cudaMemcpyHostToDevice) != cudaSuccess)
    {
        return -3;
    }
    */
    return 0;
}

int w_set_filters_inverse_nonseparable(DTYPE* filter1, DTYPE* filter2, DTYPE* filter3, DTYPE* filter4, uint len) {

    std::memcpy(c_kern_LL, filter1, len*len*sizeof(DTYPE));
    std::memcpy(c_kern_LH, filter2, len*len*sizeof(DTYPE));
    std::memcpy(c_kern_HL, filter3, len*len*sizeof(DTYPE));
    std::memcpy(c_kern_HH, filter4, len*len*sizeof(DTYPE));
   
   /*
    if (cudaMemcpyToSymbol(c_kern_LL, filter1, len*len*sizeof(DTYPE), 0, cudaMemcpyHostToDevice) != cudaSuccess
        || cudaMemcpyToSymbol(c_kern_LH, filter2, len*len*sizeof(DTYPE), 0, cudaMemcpyHostToDevice) != cudaSuccess
        || cudaMemcpyToSymbol(c_kern_HL, filter3, len*len*sizeof(DTYPE), 0, cudaMemcpyHostToDevice) != cudaSuccess
        || cudaMemcpyToSymbol(c_kern_HH, filter4, len*len*sizeof(DTYPE), 0, cudaMemcpyHostToDevice) != cudaSuccess)
    {
        return -3;
    }*/
    return 0;
}

// must be run with grid size = (Nc/2, Nr/2)  where Nr = numrows of input image
void w_kern_forward(DTYPE* img, DTYPE* c_a, DTYPE* c_h, DTYPE* c_v, DTYPE* c_d, int Nr, int Nc, int hlen) {

    int Nr_is_odd = (Nr & 1);
    int Nr2 = (Nr + Nr_is_odd)/2;
    int Nc_is_odd = (Nc & 1);
    int Nc2 = (Nc + Nc_is_odd)/2;
	
	 for(int row=0;row<Nr2;++row)
     {
	     for(int col=0;col<Nc2;++col)
	     {
	        int c, hL, hR;
	        if (hlen & 1) { // odd kernel size
	            c = hlen/2;
	            hL = c;
	            hR = c;
	        }
	        else { // even kernel size : center is shifted to the left
	            c = hlen/2 - 1;
	            hL = c;
	            hR = c+1;
	        }
	        DTYPE res_a = 0, res_h = 0, res_v = 0, res_d = 0;
	        DTYPE img_val;
	
	        // Convolution with periodic boundaries extension.
	        // The following can be sped-up by splitting into 3*3 loops, but it would be a nightmare for readability
	        for (int jy = 0; jy <= hR+hL; jy++) {
	            int idx_y = row*2 - c + jy;
	            if (idx_y < 0) idx_y += (Nr + Nr_is_odd); // if N is odd, image is virtually extended
	            // no "else if", since idx_y can be > N-1  after being incremented
	            if (idx_y > Nr-1) {
	                if ((idx_y == Nr) && (Nr_is_odd)) idx_y--; // if N is odd, repeat the right-most element
	                else idx_y -= (Nr + Nr_is_odd); // if N is odd, image is virtually extended
	            }
	
	            for (int jx = 0; jx <= hR+hL; jx++) {
	                int idx_x = col*2 - c + jx;
	                if (idx_x < 0) idx_x += (Nc + Nc_is_odd); // if N is odd, image is virtually extended
	                // no "else if", since idx_x can be > N-1  after being incremented
	                if (idx_x > Nc-1) {
	                    if ((idx_x == Nc) && (Nc_is_odd)) idx_x--; // if N is odd, repeat the right-most element
	                    else idx_x -= (Nc + Nc_is_odd); // if N is odd, image is virtually extended
	                }
	
	                img_val = img[idx_y*Nc + idx_x];
	                res_a += img_val * c_kern_LL[(hlen-1-jy)*hlen + (hlen-1 - jx)];
	                res_h += img_val * c_kern_LH[(hlen-1-jy)*hlen + (hlen-1 - jx)];
	                res_v += img_val * c_kern_HL[(hlen-1-jy)*hlen + (hlen-1 - jx)];
	                res_d += img_val * c_kern_HH[(hlen-1-jy)*hlen + (hlen-1 - jx)];
	            }
	        }
	        c_a[row* Nc2 + col] = res_a;
	        c_h[row* Nc2 + col] = res_h;
	        c_v[row* Nc2 + col] = res_v;
	        c_d[row* Nc2 + col] = res_d;
			 
		 }
	 }

/*	 
    int gidx = threadIdx.x + blockIdx.x*blockDim.x;
    int gidy = threadIdx.y + blockIdx.y*blockDim.y;

    int Nr_is_odd = (Nr & 1);
    int Nr2 = (Nr + Nr_is_odd)/2;
    int Nc_is_odd = (Nc & 1);
    int Nc2 = (Nc + Nc_is_odd)/2;

    if (gidy < Nr2 && gidx < Nc2) {
        int c, hL, hR;
        if (hlen & 1) { // odd kernel size
            c = hlen/2;
            hL = c;
            hR = c;
        }
        else { // even kernel size : center is shifted to the left
            c = hlen/2 - 1;
            hL = c;
            hR = c+1;
        }
        DTYPE res_a = 0, res_h = 0, res_v = 0, res_d = 0;
        DTYPE img_val;

        // Convolution with periodic boundaries extension.
        // The following can be sped-up by splitting into 3*3 loops, but it would be a nightmare for readability
        for (int jy = 0; jy <= hR+hL; jy++) {
            int idx_y = gidy*2 - c + jy;
            if (idx_y < 0) idx_y += (Nr + Nr_is_odd); // if N is odd, image is virtually extended
            // no "else if", since idx_y can be > N-1  after being incremented
            if (idx_y > Nr-1) {
                if ((idx_y == Nr) && (Nr_is_odd)) idx_y--; // if N is odd, repeat the right-most element
                else idx_y -= (Nr + Nr_is_odd); // if N is odd, image is virtually extended
            }

            for (int jx = 0; jx <= hR+hL; jx++) {
                int idx_x = gidx*2 - c + jx;
                if (idx_x < 0) idx_x += (Nc + Nc_is_odd); // if N is odd, image is virtually extended
                // no "else if", since idx_x can be > N-1  after being incremented
                if (idx_x > Nc-1) {
                    if ((idx_x == Nc) && (Nc_is_odd)) idx_x--; // if N is odd, repeat the right-most element
                    else idx_x -= (Nc + Nc_is_odd); // if N is odd, image is virtually extended
                }

                img_val = img[idx_y*Nc + idx_x];
                res_a += img_val * c_kern_LL[(hlen-1-jy)*hlen + (hlen-1 - jx)];
                res_h += img_val * c_kern_LH[(hlen-1-jy)*hlen + (hlen-1 - jx)];
                res_v += img_val * c_kern_HL[(hlen-1-jy)*hlen + (hlen-1 - jx)];
                res_d += img_val * c_kern_HH[(hlen-1-jy)*hlen + (hlen-1 - jx)];
            }
        }
        c_a[gidy* Nc2 + gidx] = res_a;
        c_h[gidy* Nc2 + gidx] = res_h;
        c_v[gidy* Nc2 + gidx] = res_v;
        c_d[gidy* Nc2 + gidx] = res_d;
    }
*/ 
}




// must be run with grid size = (2*Nr, 2*Nc) ; Nr = numrows of input
void w_kern_inverse(DTYPE* img, DTYPE* c_a, DTYPE* c_h, DTYPE* c_v, DTYPE* c_d, int Nr, int Nc, int Nr2, int Nc2, int hlen) {

	 for(int row=0;row<Nr2;++row)
     {
	     for(int col=0;col<Nc2;++col)
	     {
			int internal_row = row; 
			int internal_col = col; 
	        int c, hL, hR;
	        int hlen2 = hlen/2; // Convolutions with even/odd indices of the kernels
	        if (hlen2 & 1) { // odd half-kernel size
	            c = hlen2/2;
	            hL = c;
	            hR = c;
	        }
	        else { // even half-kernel size : center is shifted to the RIGHT for reconstruction.
	            c = hlen2/2 - 0;
	            hL = c;
	            hR = c-1;
	            // virtual id for shift
	            // TODO : for the very first convolution (on the edges), this is not exactly accurate (?)
	            internal_col += 1;
	            internal_row += 1;
	        }
	        int jy1 = c - internal_row/2;
	        int jy2 = Nr - 1 - internal_row/2 + c;
	        int jx1 = c - internal_col/2;
	        int jx2 = Nc - 1 - internal_col/2 + c;
	
	        // There are 4 threads/coeff index. Each thread will do a convolution with the even/odd indices of the kernels along each dimension.
	        int offset_x = 1-(internal_col & 1);
	        int offset_y = 1-(internal_row & 1);
	
	        DTYPE res_a = 0, res_h = 0, res_v = 0, res_d = 0;
	        for (int jy = 0; jy <= hR+hL; jy++) {
	            int idx_y = internal_row/2 - c + jy;
	            if (jy < jy1) idx_y += Nr;
	            if (jy > jy2) idx_y -= Nr;
	            for (int jx = 0; jx <= hR+hL; jx++) {
	                int idx_x = internal_col/2 - c + jx;
	                if (jx < jx1) idx_x += Nc;
	                if (jx > jx2) idx_x -= Nc;
	
	                res_a += c_a[idx_y*Nc + idx_x] * c_kern_LL[(hlen-1- (2*jy + offset_y))*hlen + (hlen-1 - (2*jx + offset_x))];
	                res_h += c_h[idx_y*Nc + idx_x] * c_kern_LH[(hlen-1- (2*jy + offset_y))*hlen + (hlen-1 - (2*jx + offset_x))];
	                res_v += c_v[idx_y*Nc + idx_x] * c_kern_HL[(hlen-1- (2*jy + offset_y))*hlen + (hlen-1 - (2*jx + offset_x))];
	                res_d += c_d[idx_y*Nc + idx_x] * c_kern_HH[(hlen-1- (2*jy + offset_y))*hlen + (hlen-1 - (2*jx + offset_x))];
	            }
	        }
	        if ((hlen2 & 1) == 1) img[internal_row * Nc2 + internal_col] = res_a + res_h + res_v + res_d;
	        else img[(internal_row-1) * Nc2 + (internal_col-1)] = res_a + res_h + res_v + res_d;
	
		 }
	 }

/*
    int gidx = threadIdx.x + blockIdx.x*blockDim.x;
    int gidy = threadIdx.y + blockIdx.y*blockDim.y;
    if (gidy < Nr2 && gidx < Nc2) {

        int c, hL, hR;
        int hlen2 = hlen/2; // Convolutions with even/odd indices of the kernels
        if (hlen2 & 1) { // odd half-kernel size
            c = hlen2/2;
            hL = c;
            hR = c;
        }
        else { // even half-kernel size : center is shifted to the RIGHT for reconstruction.
            c = hlen2/2 - 0;
            hL = c;
            hR = c-1;
            // virtual id for shift
            // TODO : for the very first convolution (on the edges), this is not exactly accurate (?)
            gidx += 1;
            gidy += 1;
        }
        int jy1 = c - gidy/2;
        int jy2 = Nr - 1 - gidy/2 + c;
        int jx1 = c - gidx/2;
        int jx2 = Nc - 1 - gidx/2 + c;

        // There are 4 threads/coeff index. Each thread will do a convolution with the even/odd indices of the kernels along each dimension.
        int offset_x = 1-(gidx & 1);
        int offset_y = 1-(gidy & 1);

        DTYPE res_a = 0, res_h = 0, res_v = 0, res_d = 0;
        for (int jy = 0; jy <= hR+hL; jy++) {
            int idx_y = gidy/2 - c + jy;
            if (jy < jy1) idx_y += Nr;
            if (jy > jy2) idx_y -= Nr;
            for (int jx = 0; jx <= hR+hL; jx++) {
                int idx_x = gidx/2 - c + jx;
                if (jx < jx1) idx_x += Nc;
                if (jx > jx2) idx_x -= Nc;

                res_a += c_a[idx_y*Nc + idx_x] * c_kern_LL[(hlen-1- (2*jy + offset_y))*hlen + (hlen-1 - (2*jx + offset_x))];
                res_h += c_h[idx_y*Nc + idx_x] * c_kern_LH[(hlen-1- (2*jy + offset_y))*hlen + (hlen-1 - (2*jx + offset_x))];
                res_v += c_v[idx_y*Nc + idx_x] * c_kern_HL[(hlen-1- (2*jy + offset_y))*hlen + (hlen-1 - (2*jx + offset_x))];
                res_d += c_d[idx_y*Nc + idx_x] * c_kern_HH[(hlen-1- (2*jy + offset_y))*hlen + (hlen-1 - (2*jx + offset_x))];
            }
        }
        if ((hlen2 & 1) == 1) img[gidy * Nc2 + gidx] = res_a + res_h + res_v + res_d;
        else img[(gidy-1) * Nc2 + (gidx-1)] = res_a + res_h + res_v + res_d;
    }
*/ 
}







int w_forward(DTYPE* d_image, wavelet_coeff_t& d_coeffs, DTYPE* d_tmp, w_info winfos) {

    int Nr = winfos.Nr, Nc = winfos.Nc, levels = winfos.nlevels, hlen = winfos.hlen;
    int Nc2 = Nc, Nr2 = Nr;
    int Nc2_old = Nc2, Nr2_old = Nr2;
    w_div2(&Nr2); w_div2(&Nc2);
    DTYPE* d_tmp1, *d_tmp2;
    d_tmp1 = d_coeffs[0].data();
    d_tmp2 = d_tmp;

 std::cout << "Forward nonseparable " << std::endl;
    
		for(auto filter_coeff : c_kern_LL)
		{
			std::cout << filter_coeff << '\t';
		}
        std::cout << std::endl;
		for(auto filter_coeff : c_kern_LH)
		{
			std::cout << filter_coeff << '\t';
		}
        std::cout << std::endl;
		for(auto filter_coeff : c_kern_HL)
		{
			std::cout << filter_coeff << '\t';
		}
        std::cout << std::endl;
		for(auto filter_coeff : c_kern_HH)
		{
			std::cout << filter_coeff << '\t';
		}
        std::cout << std::endl;



    // First level
    w_kern_forward(d_image, d_coeffs[0].data(), d_coeffs[1].data(), d_coeffs[2].data(), d_coeffs[3].data(), Nr, Nc, hlen);


    std::cout << "Forward nonseparable 1 pass coeff0" << std::endl;

    for(int i=0;i<10;++i)
	{
		std::cout << (d_coeffs[0])[i] << '\t';
	}
        std::cout << std::endl;
    for (int i=1; i < levels; i++) {
        Nr2_old = Nr2; Nc2_old = Nc2;
        w_div2(&Nr2); w_div2(&Nc2);
        w_kern_forward(d_tmp1, d_tmp2, d_coeffs[3*i+1].data(), d_coeffs[3*i+2].data(), d_coeffs[3*i+3].data(), Nr2_old, Nc2_old, hlen);
        w_swap_ptr(&d_tmp1, &d_tmp2);

    std::cout << "Forward nonseparable 1 pass coeff" << std::endl;

    for(int j=0;j<10;++j)
	{
		std::cout << (d_coeffs[3*i+1])[j] << '\t';
	}
        
    }
    if ((levels > 1) && ((levels & 1) == 0))
    {
		std::memcpy(d_coeffs[0].data(), d_tmp, Nr2*Nc2*sizeof(DTYPE));
		
		// cudaMemcpy(d_coeffs[0], d_tmp, Nr2*Nc2*sizeof(DTYPE), cudaMemcpyDeviceToDevice);
    }
    return 0;

/*
    int Nr = winfos.Nr, Nc = winfos.Nc, levels = winfos.nlevels, hlen = winfos.hlen;
    int tpb = 16; // TODO : tune for max perfs.
    int Nc2 = Nc, Nr2 = Nr;
    int Nc2_old = Nc2, Nr2_old = Nr2;
    w_div2(&Nr2); w_div2(&Nc2);
    DTYPE* d_tmp1, *d_tmp2;
    d_tmp1 = d_coeffs[0];
    d_tmp2 = d_tmp;

    // First level
    dim3 n_blocks = dim3(w_iDivUp(Nc2, tpb), w_iDivUp(Nr2, tpb), 1);
    dim3 n_threads_per_block = dim3(tpb, tpb, 1);
    w_kern_forward<<<n_blocks, n_threads_per_block>>>(d_image, d_coeffs[0], d_coeffs[1], d_coeffs[2], d_coeffs[3], Nr, Nc, hlen);

    for (int i=1; i < levels; i++) {
        Nr2_old = Nr2; Nc2_old = Nc2;
        w_div2(&Nr2); w_div2(&Nc2);
        n_blocks = dim3(w_iDivUp(Nc2, tpb), w_iDivUp(Nr2, tpb), 1);
        w_kern_forward<<<n_blocks, n_threads_per_block>>>(d_tmp1, d_tmp2, d_coeffs[3*i+1], d_coeffs[3*i+2], d_coeffs[3*i+3], Nr2_old, Nc2_old, hlen);
        w_swap_ptr(&d_tmp1, &d_tmp2);
    }
    if ((levels > 1) && ((levels & 1) == 0)) cudaMemcpy(d_coeffs[0], d_tmp, Nr2*Nc2*sizeof(DTYPE), cudaMemcpyDeviceToDevice);
    return 0;
*/ 
}


int w_inverse(DTYPE* d_image, wavelet_coeff_t& d_coeffs, DTYPE* d_tmp, w_info winfos) {

    int Nr = winfos.Nr, Nc = winfos.Nc, levels = winfos.nlevels, hlen = winfos.hlen;
     // Table of sizes. FIXME: consider adding this in the w_info structure
    int tNr[levels+1]; tNr[0] = Nr;
    int tNc[levels+1]; tNc[0] = Nc;
    for (int i = 1; i <= levels; i++) {
        tNr[i] = tNr[i-1];
        tNc[i] = tNc[i-1];
        w_div2(tNr + i);
        w_div2(tNc + i);
    }

    DTYPE* d_tmp1, *d_tmp2;
    d_tmp1 = d_coeffs[0].data();
    d_tmp2 = d_tmp;

    for (int i = levels-1; i >= 1; i--) {
        w_kern_inverse(d_tmp2, d_tmp1, d_coeffs[3*i+1].data(), d_coeffs[3*i+2].data(), d_coeffs[3*i+3].data(), tNr[i+1], tNc[i+1], tNr[i], tNc[i], hlen);
        w_swap_ptr(&d_tmp1, &d_tmp2);
    }
    if ((levels > 1) && ((levels & 1) == 0))
    {
		std::memcpy(d_coeffs[0].data(), d_tmp, tNr[1]*tNc[1]*sizeof(DTYPE));
	  //	 cudaMemcpy(d_coeffs[0], d_tmp, tNr[1]*tNc[1]*sizeof(DTYPE), cudaMemcpyDeviceToDevice);
    }
    // First level
    w_kern_inverse(d_image, d_coeffs[0].data(), d_coeffs[1].data(), d_coeffs[2].data(), d_coeffs[3].data(), tNr[1], tNc[1], Nr, Nc, hlen);

    return 0;


/*
    int Nr = winfos.Nr, Nc = winfos.Nc, levels = winfos.nlevels, hlen = winfos.hlen;
     // Table of sizes. FIXME: consider adding this in the w_info structure
    int tNr[levels+1]; tNr[0] = Nr;
    int tNc[levels+1]; tNc[0] = Nc;
    for (int i = 1; i <= levels; i++) {
        tNr[i] = tNr[i-1];
        tNc[i] = tNc[i-1];
        w_div2(tNr + i);
        w_div2(tNc + i);
    }
    int tpb = 16; // TODO : tune for max perfs.

    DTYPE* d_tmp1, *d_tmp2;
    d_tmp1 = d_coeffs[0];
    d_tmp2 = d_tmp;

    dim3 n_threads_per_block = dim3(tpb, tpb, 1);
    dim3 n_blocks;
    for (int i = levels-1; i >= 1; i--) {
        n_blocks = dim3(w_iDivUp(tNc[i], tpb), w_iDivUp(tNr[i], tpb), 1);
        w_kern_inverse<<<n_blocks, n_threads_per_block>>>(d_tmp2, d_tmp1, d_coeffs[3*i+1], d_coeffs[3*i+2], d_coeffs[3*i+3], tNr[i+1], tNc[i+1], tNr[i], tNc[i], hlen);
        w_swap_ptr(&d_tmp1, &d_tmp2);
    }
    if ((levels > 1) && ((levels & 1) == 0)) cudaMemcpy(d_coeffs[0], d_tmp, tNr[1]*tNc[1]*sizeof(DTYPE), cudaMemcpyDeviceToDevice);
    // First level
    n_blocks = dim3(w_iDivUp(Nc, tpb), w_iDivUp(Nr, tpb), 1);
    w_kern_inverse<<<n_blocks, n_threads_per_block>>>(d_image, d_coeffs[0], d_coeffs[1], d_coeffs[2], d_coeffs[3], tNr[1], tNc[1], Nr, Nc, hlen);

    return 0;
*/
}





/// ----------------------------------------------------------------------------
/// -------------------------   Undecimated DWT --------------------------------
/// ----------------------------------------------------------------------------



// must be run with grid size = (Nc, Nr)  where Nr = numrows of input image
void w_kern_forward_swt(DTYPE* img, DTYPE* c_a, DTYPE* c_h, DTYPE* c_v, DTYPE* c_d, int Nr, int Nc, int hlen, int level) {

	 for(int row=0;row<Nr;++row)
     {
	     for(int col=0;col<Nc;++col)
	     {
	        int factor = 1 << (level - 1);
	        int c, hL, hR;
	        if (hlen & 1) { // odd kernel size
	            c = hlen/2;
	            hL = c;
	            hR = c;
	        }
	        else { // even kernel size : center is shifted to the left
	            c = hlen/2 - 1;
	            hL = c;
	            hR = c+1;
	        }
	
	        c *= factor;
	        int jx1 = c - col;
	        int jx2 = Nc - 1 - col + c;
	        int jy1 = c - row;
	        int jy2 = Nr - 1 - row + c;
	        DTYPE res_a = 0, res_h = 0, res_v = 0, res_d = 0;
	        DTYPE img_val;
	
	        // Convolution with periodic boundaries extension.
	        // The filters are 2-upsampled at each level : [h0, h1, h2, h3] --> [h0, 0, h1, 0, h2, 0, h3, 0]
	        for (int jy = 0; jy <= hR+hL; jy++) {
	            int idx_y = row - c + factor*jy;
	            if (factor*jy < jy1) idx_y += Nr;
	            if (factor*jy > jy2) idx_y -= Nr;
	            for (int jx = 0; jx <= hR+hL; jx++) {
	                int idx_x = col + jx*factor - c;
	                if (factor*jx < jx1) idx_x += Nc;
	                if (factor*jx > jx2) idx_x -= Nc;
	
	                img_val = img[idx_y*Nc + idx_x];
	                res_a += img_val * c_kern_LL[(hlen-1-jy)*hlen + (hlen-1 - jx)];
	                res_h += img_val * c_kern_LH[(hlen-1-jy)*hlen + (hlen-1 - jx)];
	                res_v += img_val * c_kern_HL[(hlen-1-jy)*hlen + (hlen-1 - jx)];
	                res_d += img_val * c_kern_HH[(hlen-1-jy)*hlen + (hlen-1 - jx)];
	
	            }
	        }
	        c_a[row* Nc + col] = res_a;
	        c_h[row* Nc + col] = res_h;
	        c_v[row* Nc + col] = res_v;
	        c_d[row* Nc + col] = res_d;
  
		 }
	 }


/*
    int gidx = threadIdx.x + blockIdx.x*blockDim.x;
    int gidy = threadIdx.y + blockIdx.y*blockDim.y;
    if (gidy < Nr && gidx < Nc) {

        int factor = 1 << (level - 1);
        int c, hL, hR;
        if (hlen & 1) { // odd kernel size
            c = hlen/2;
            hL = c;
            hR = c;
        }
        else { // even kernel size : center is shifted to the left
            c = hlen/2 - 1;
            hL = c;
            hR = c+1;
        }

        c *= factor;
        int jx1 = c - gidx;
        int jx2 = Nc - 1 - gidx + c;
        int jy1 = c - gidy;
        int jy2 = Nr - 1 - gidy + c;
        DTYPE res_a = 0, res_h = 0, res_v = 0, res_d = 0;
        DTYPE img_val;

        // Convolution with periodic boundaries extension.
        // The filters are 2-upsampled at each level : [h0, h1, h2, h3] --> [h0, 0, h1, 0, h2, 0, h3, 0]
        for (int jy = 0; jy <= hR+hL; jy++) {
            int idx_y = gidy - c + factor*jy;
            if (factor*jy < jy1) idx_y += Nr;
            if (factor*jy > jy2) idx_y -= Nr;
            for (int jx = 0; jx <= hR+hL; jx++) {
                int idx_x = gidx + jx*factor - c;
                if (factor*jx < jx1) idx_x += Nc;
                if (factor*jx > jx2) idx_x -= Nc;

                img_val = img[idx_y*Nc + idx_x];
                res_a += img_val * c_kern_LL[(hlen-1-jy)*hlen + (hlen-1 - jx)];
                res_h += img_val * c_kern_LH[(hlen-1-jy)*hlen + (hlen-1 - jx)];
                res_v += img_val * c_kern_HL[(hlen-1-jy)*hlen + (hlen-1 - jx)];
                res_d += img_val * c_kern_HH[(hlen-1-jy)*hlen + (hlen-1 - jx)];

            }
        }
        c_a[gidy* Nc + gidx] = res_a;
        c_h[gidy* Nc + gidx] = res_h;
        c_v[gidy* Nc + gidx] = res_v;
        c_d[gidy* Nc + gidx] = res_d;
    }
*/
}




// must be run with grid size = (2*Nr, 2*Nc) ; Nr = numrows of input
void w_kern_inverse_swt(DTYPE* img, DTYPE* c_a, DTYPE* c_h, DTYPE* c_v, DTYPE* c_d, int Nr, int Nc, int hlen, int level) {

	 for(int row=0;row<Nr;++row)
     {
	     for(int col=0;col<Nc;++col)
	     {
	        int factor = 1 << (level - 1);
	        int c, hL, hR;
	        if (hlen & 1) { // odd half-kernel size
	            c = hlen/2;
	            hL = c;
	            hR = c;
	        }
	        else { // even half-kernel size : center is shifted to the RIGHT for reconstruction.
	            c = hlen/2 - 0;
	            hL = c;
	            hR = c-1;
	        }
	        c *= factor;
	        int jy1 = c - row;
	        int jy2 = Nr - 1 - row + c;
	        int jx1 = c - col;
	        int jx2 = Nc - 1 - col + c;
	
	        DTYPE res_a = 0, res_h = 0, res_v = 0, res_d = 0;
	        for (int jy = 0; jy <= hR+hL; jy++) {
	            int idx_y = row - c + jy*factor;
	            if (factor*jy < jy1) idx_y += Nr;
	            if (factor*jy > jy2) idx_y -= Nr;
	            for (int jx = 0; jx <= hR+hL; jx++) {
	                int idx_x = col - c + jx*factor;
	                if (factor*jx < jx1) idx_x += Nc;
	                if (factor*jx > jx2) idx_x -= Nc;
	
	                res_a += c_a[idx_y*Nc + idx_x] * c_kern_LL[(hlen-1-jy)*hlen + (hlen-1 - jx)]/4;
	                res_h += c_h[idx_y*Nc + idx_x] * c_kern_LH[(hlen-1-jy)*hlen + (hlen-1 - jx)]/4;
	                res_v += c_v[idx_y*Nc + idx_x] * c_kern_HL[(hlen-1-jy)*hlen + (hlen-1 - jx)]/4;
	                res_d += c_d[idx_y*Nc + idx_x] * c_kern_HH[(hlen-1-jy)*hlen + (hlen-1 - jx)]/4;
	            }
	        }
	        img[row * Nc + col] = res_a + res_h + res_v + res_d;
	
		 }
	 }

/*
    int gidx = threadIdx.x + blockIdx.x*blockDim.x;
    int gidy = threadIdx.y + blockIdx.y*blockDim.y;
    if (gidy < Nr && gidx < Nc) {

        int factor = 1 << (level - 1);
        int c, hL, hR;
        if (hlen & 1) { // odd half-kernel size
            c = hlen/2;
            hL = c;
            hR = c;
        }
        else { // even half-kernel size : center is shifted to the RIGHT for reconstruction.
            c = hlen/2 - 0;
            hL = c;
            hR = c-1;
        }
        c *= factor;
        int jy1 = c - gidy;
        int jy2 = Nr - 1 - gidy + c;
        int jx1 = c - gidx;
        int jx2 = Nc - 1 - gidx + c;

        DTYPE res_a = 0, res_h = 0, res_v = 0, res_d = 0;
        for (int jy = 0; jy <= hR+hL; jy++) {
            int idx_y = gidy - c + jy*factor;
            if (factor*jy < jy1) idx_y += Nr;
            if (factor*jy > jy2) idx_y -= Nr;
            for (int jx = 0; jx <= hR+hL; jx++) {
                int idx_x = gidx - c + jx*factor;
                if (factor*jx < jx1) idx_x += Nc;
                if (factor*jx > jx2) idx_x -= Nc;

                res_a += c_a[idx_y*Nc + idx_x] * c_kern_LL[(hlen-1-jy)*hlen + (hlen-1 - jx)]/4;
                res_h += c_h[idx_y*Nc + idx_x] * c_kern_LH[(hlen-1-jy)*hlen + (hlen-1 - jx)]/4;
                res_v += c_v[idx_y*Nc + idx_x] * c_kern_HL[(hlen-1-jy)*hlen + (hlen-1 - jx)]/4;
                res_d += c_d[idx_y*Nc + idx_x] * c_kern_HH[(hlen-1-jy)*hlen + (hlen-1 - jx)]/4;
            }
        }
        img[gidy * Nc + gidx] = res_a + res_h + res_v + res_d;
    }
*/ 
}






int w_forward_swt(DTYPE* d_image, wavelet_coeff_t& d_coeffs, DTYPE* d_tmp, w_info winfos) {

    int Nr = winfos.Nr, Nc = winfos.Nc, levels = winfos.nlevels, hlen = winfos.hlen;

    DTYPE* d_tmp1, *d_tmp2;
    d_tmp1 = d_coeffs[0].data();
    d_tmp2 = d_tmp;

    // First level
    w_kern_forward_swt(d_image, d_coeffs[0].data(), d_coeffs[1].data(), d_coeffs[2].data(), d_coeffs[3].data(), Nr, Nc, hlen, 1);
    for (int i=1; i < levels; i++) {
        w_kern_forward_swt(d_tmp1, d_tmp2, d_coeffs[3*i+1].data(), d_coeffs[3*i+2].data(), d_coeffs[3*i+3].data(), Nr, Nc, hlen, i+1);
        w_swap_ptr(&d_tmp1, &d_tmp2);
    }
    if ((levels & 1) == 0)
    {
		std::memcpy(d_coeffs[0].data(), d_tmp, Nr*Nc*sizeof(DTYPE));
		 //cudaMemcpy(d_coeffs[0], d_tmp, Nr*Nc*sizeof(DTYPE), cudaMemcpyDeviceToDevice);
    }
    return 0;

/*
    int Nr = winfos.Nr, Nc = winfos.Nc, levels = winfos.nlevels, hlen = winfos.hlen;

    DTYPE* d_tmp1, *d_tmp2;
    d_tmp1 = d_coeffs[0];
    d_tmp2 = d_tmp;

    // First level
    int tpb = 16; // TODO : tune for max perfs.
    dim3 n_blocks = dim3(w_iDivUp(Nc, tpb), w_iDivUp(Nr, tpb), 1);
    dim3 n_threads_per_block = dim3(tpb, tpb, 1);
    w_kern_forward_swt<<<n_blocks, n_threads_per_block>>>(d_image, d_coeffs[0], d_coeffs[1], d_coeffs[2], d_coeffs[3], Nr, Nc, hlen, 1);
    for (int i=1; i < levels; i++) {
        w_kern_forward_swt<<<n_blocks, n_threads_per_block>>>(d_tmp1, d_tmp2, d_coeffs[3*i+1], d_coeffs[3*i+2], d_coeffs[3*i+3], Nr, Nc, hlen, i+1);
        w_swap_ptr(&d_tmp1, &d_tmp2);
    }
    if ((levels & 1) == 0) cudaMemcpy(d_coeffs[0], d_tmp, Nr*Nc*sizeof(DTYPE), cudaMemcpyDeviceToDevice);
    return 0;
*/
}



int w_inverse_swt(DTYPE* d_image, wavelet_coeff_t& d_coeffs, DTYPE* d_tmp, w_info winfos) {

    int Nr = winfos.Nr, Nc = winfos.Nc, levels = winfos.nlevels, hlen = winfos.hlen;
    DTYPE* d_tmp1, *d_tmp2;
    d_tmp1 = d_coeffs[0].data();
    d_tmp2 = d_tmp;

    for (int i = levels-1; i >= 1; i--) {
        w_kern_inverse_swt(d_tmp2, d_tmp1, d_coeffs[3*i+1].data(), d_coeffs[3*i+2].data(), d_coeffs[3*i+3].data(), Nr, Nc, hlen, i+1);
        w_swap_ptr(&d_tmp1, &d_tmp2);
    }
    if ((levels & 1) == 0)
    {
	   std::memcpy(d_coeffs[0].data(), d_tmp, Nr*Nc*sizeof(DTYPE));	
	 //	 cudaMemcpy(d_coeffs[0], d_tmp, Nr*Nc*sizeof(DTYPE), cudaMemcpyDeviceToDevice);
    }
    // First scale
    w_kern_inverse_swt(d_image, d_coeffs[0].data(), d_coeffs[1].data(), d_coeffs[2].data(), d_coeffs[3].data(), Nr, Nc, hlen, 1);

    return 0;
   
/*   
    int Nr = winfos.Nr, Nc = winfos.Nc, levels = winfos.nlevels, hlen = winfos.hlen;
    DTYPE* d_tmp1, *d_tmp2;
    d_tmp1 = d_coeffs[0];
    d_tmp2 = d_tmp;

    int tpb = 16; // TODO : tune for max perfs.
    dim3 n_threads_per_block = dim3(tpb, tpb, 1);
    dim3 n_blocks = dim3(w_iDivUp(Nc, tpb), w_iDivUp(Nr, tpb), 1);
    for (int i = levels-1; i >= 1; i--) {
        w_kern_inverse_swt<<<n_blocks, n_threads_per_block>>>(d_tmp2, d_tmp1, d_coeffs[3*i+1], d_coeffs[3*i+2], d_coeffs[3*i+3], Nr, Nc, hlen, i+1);
        w_swap_ptr(&d_tmp1, &d_tmp2);
    }
    if ((levels & 1) == 0) cudaMemcpy(d_coeffs[0], d_tmp, Nr*Nc*sizeof(DTYPE), cudaMemcpyDeviceToDevice);
    // First scale
    n_blocks = dim3(w_iDivUp(Nc, tpb), w_iDivUp(Nr, tpb), 1);
    w_kern_inverse_swt<<<n_blocks, n_threads_per_block>>>(d_image, d_coeffs[0], d_coeffs[1], d_coeffs[2], d_coeffs[3], Nr, Nc, hlen, 1);

    return 0;
*/
}


nonseparable_wavelet_transform::nonseparable_wavelet_transform(w_info winfo):winfos(winfo)
{
	  tmp.resize(2*winfos.Nr*winfos.Nc);
}
	
nonseparable_wavelet_transform::~nonseparable_wavelet_transform()
{
	
}
	
int nonseparable_wavelet_transform::set_filters(const char* wname, int do_swt)
{
	 
    int hlen = 0;
    DTYPE* f1_l; // 1D lowpass
    DTYPE* f1_h; // 1D highpass
    DTYPE* f2_a, *f2_h, *f2_v, *f2_d; // 2D filters

    // Haar filters has specific kernels
    if (!do_swt) {
        if ((!strcasecmp(wname, "haar")) || (!strcasecmp(wname, "db1")) || (!strcasecmp(wname, "bior1.1")) || (!strcasecmp(wname, "rbior1.1"))) {
            return 2;
        }
    }

    // Browse available filters (see filters.h)
    int filter_index;
    for (filter_index = 0; filter_index < 72; filter_index++) {
        if (!strcasecmp(wname, all_filters[filter_index].wname)) {
            hlen = all_filters[filter_index].hlen;
                f1_l = all_filters[filter_index].f_l;
                f1_h = all_filters[filter_index].f_h;
  
            break;
        }
    }
    if (hlen == 0) {
        printf("ERROR: w_compute_filters(): unknown filter %s\n", wname);
        return -2;
    }

    // Create the separable 2D filters
    f2_a = w_outer(f1_l, f1_l, hlen);
    f2_h = w_outer(f1_l, f1_h, hlen); // CHECKME
    f2_v = w_outer(f1_h, f1_l, hlen);
    f2_d = w_outer(f1_h, f1_h, hlen);

    // Copy the filters to device constant memory
    c_kern_LL.resize(hlen*hlen);
    c_kern_LH.resize(hlen*hlen);
    c_kern_HL.resize(hlen*hlen);
    c_kern_HH.resize(hlen*hlen);
    std::memcpy(c_kern_LL.data(), f2_a, hlen*hlen*sizeof(DTYPE));
    std::memcpy(c_kern_LH.data(), f2_h, hlen*hlen*sizeof(DTYPE));
    std::memcpy(c_kern_HL.data(), f2_v, hlen*hlen*sizeof(DTYPE));
    std::memcpy(c_kern_HH.data(), f2_d, hlen*hlen*sizeof(DTYPE));

    f1_l = all_filters[filter_index].i_l;
    f1_h = all_filters[filter_index].i_h;

    f2_a = w_outer(f1_l, f1_l, hlen);
    f2_h = w_outer(f1_l, f1_h, hlen); // CHECKME
    f2_v = w_outer(f1_h, f1_l, hlen);
    f2_d = w_outer(f1_h, f1_h, hlen);

    // Copy the filters to device constant memory
    c_kern_ILL.resize(hlen*hlen);
    c_kern_ILH.resize(hlen*hlen);
    c_kern_IHL.resize(hlen*hlen);
    c_kern_IHH.resize(hlen*hlen);
    std::memcpy(c_kern_ILL.data(), f2_a, hlen*hlen*sizeof(DTYPE));
    std::memcpy(c_kern_ILH.data(), f2_h, hlen*hlen*sizeof(DTYPE));
    std::memcpy(c_kern_IHL.data(), f2_v, hlen*hlen*sizeof(DTYPE));
    std::memcpy(c_kern_IHH.data(), f2_d, hlen*hlen*sizeof(DTYPE));

    winfos.hlen = hlen;
     
   /* 
    cudaMemcpyToSymbol(c_kern_LL, f2_a, hlen*hlen*sizeof(DTYPE), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(c_kern_LH, f2_h, hlen*hlen*sizeof(DTYPE), 0, cudaMemcpyHostToDevice); // CHECKME
    cudaMemcpyToSymbol(c_kern_HL, f2_v, hlen*hlen*sizeof(DTYPE), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(c_kern_HH, f2_d, hlen*hlen*sizeof(DTYPE), 0, cudaMemcpyHostToDevice);
*/
    return hlen;
}
    
int nonseparable_wavelet_transform::set_filters_forward(DTYPE* filter1, DTYPE* filter2, DTYPE* filter3, DTYPE* filter4, uint len)
{


    c_kern_LL.resize(len*len);
    c_kern_LH.resize(len*len);
    c_kern_HL.resize(len*len);
    c_kern_HH.resize(len*len);

    std::memcpy(c_kern_LL.data(), filter1, len*len*sizeof(DTYPE));
    std::memcpy(c_kern_LH.data(), filter2, len*len*sizeof(DTYPE));
    std::memcpy(c_kern_HL.data(), filter3, len*len*sizeof(DTYPE));
    std::memcpy(c_kern_HH.data(), filter4, len*len*sizeof(DTYPE));

    winfos.hlen = len;
    
    return len;

}

int nonseparable_wavelet_transform::set_filters_inverse(DTYPE* filter1, DTYPE* filter2, DTYPE* filter3, DTYPE* filter4, uint len)
{

    c_kern_ILL.resize(len*len);
	c_kern_ILH.resize(len*len);
    c_kern_IHL.resize(len*len);
    c_kern_IHH.resize(len*len);

    std::memcpy(c_kern_ILL.data(), filter1, len*len*sizeof(DTYPE));
    std::memcpy(c_kern_ILH.data(), filter2, len*len*sizeof(DTYPE));
    std::memcpy(c_kern_IHL.data(), filter3, len*len*sizeof(DTYPE));
    std::memcpy(c_kern_IHH.data(), filter4, len*len*sizeof(DTYPE));

    winfos.hlen = len;
    
    return len;
	
}
    
int nonseparable_wavelet_transform::forward_2D(DTYPE* d_image, wavelet_coeff_t& d_coeffs)
{
	
    int Nr = winfos.Nr, Nc = winfos.Nc, levels = winfos.nlevels, hlen = winfos.hlen;
    int Nc2 = Nc, Nr2 = Nr;
    int Nc2_old = Nc2, Nr2_old = Nr2;
    w_div2(&Nr2); w_div2(&Nc2);
    DTYPE* d_tmp1, *d_tmp2;
    d_tmp1 = d_coeffs[0].data();
    d_tmp2 = tmp.data();
/*
 std::cout << "Forward nonseparable " << std::endl;
    
		for(auto filter_coeff : c_kern_LL)
		{
			std::cout << filter_coeff << '\t';
		}
        std::cout << std::endl;
		for(auto filter_coeff : c_kern_LH)
		{
			std::cout << filter_coeff << '\t';
		}
        std::cout << std::endl;
		for(auto filter_coeff : c_kern_HL)
		{
			std::cout << filter_coeff << '\t';
		}
        std::cout << std::endl;
		for(auto filter_coeff : c_kern_HH)
		{
			std::cout << filter_coeff << '\t';
		}
        std::cout << std::endl;

*/

    // First level
    forward(d_image, d_coeffs[0].data(), d_coeffs[1].data(), d_coeffs[2].data(), d_coeffs[3].data(), Nr, Nc, hlen);


   /* std::cout << "Forward nonseparable 1 pass coeff0" << std::endl;

    for(int i=0;i<10;++i)
	{
		std::cout << (d_coeffs[0])[i] << '\t';
	}
        std::cout << std::endl;*/
    for (int i=1; i < levels; i++) {
        Nr2_old = Nr2; Nc2_old = Nc2;
        w_div2(&Nr2); w_div2(&Nc2);
       forward(d_tmp1, d_tmp2, d_coeffs[3*i+1].data(), d_coeffs[3*i+2].data(), d_coeffs[3*i+3].data(), Nr2_old, Nc2_old, hlen);
        w_swap_ptr(&d_tmp1, &d_tmp2);

  /*  std::cout << "Forward nonseparable 1 pass coeff" << std::endl;

    for(int j=0;j<10;++j)
	{
		std::cout << (d_coeffs[3*i+1])[j] << '\t';
	}*/
        
    }
    if ((levels > 1) && ((levels & 1) == 0))
    {
		std::memcpy(d_coeffs[0].data(), tmp.data(), Nr2*Nc2*sizeof(DTYPE));
		
		// cudaMemcpy(d_coeffs[0], d_tmp, Nr2*Nc2*sizeof(DTYPE), cudaMemcpyDeviceToDevice);
    }
    return 0;
}

int nonseparable_wavelet_transform::inverse_2D(DTYPE* d_image, wavelet_coeff_t& d_coeffs)
{
	int Nr = winfos.Nr, Nc = winfos.Nc, levels = winfos.nlevels, hlen = winfos.hlen;
     // Table of sizes. FIXME: consider adding this in the w_info structure
    int tNr[levels+1]; tNr[0] = Nr;
    int tNc[levels+1]; tNc[0] = Nc;
    for (int i = 1; i <= levels; i++) {
        tNr[i] = tNr[i-1];
        tNc[i] = tNc[i-1];
        w_div2(tNr + i);
        w_div2(tNc + i);
    }

    DTYPE* d_tmp1, *d_tmp2;
    d_tmp1 = d_coeffs[0].data();
    d_tmp2 = tmp.data();

    for (int i = levels-1; i >= 1; i--) {
        inverse(d_tmp2, d_tmp1, d_coeffs[3*i+1].data(), d_coeffs[3*i+2].data(), d_coeffs[3*i+3].data(), tNr[i+1], tNc[i+1], tNr[i], tNc[i], hlen);
        w_swap_ptr(&d_tmp1, &d_tmp2);
    }
    if ((levels > 1) && ((levels & 1) == 0))
    {
		std::memcpy(d_coeffs[0].data(), tmp.data(), tNr[1]*tNc[1]*sizeof(DTYPE));
	  //	 cudaMemcpy(d_coeffs[0], d_tmp, tNr[1]*tNc[1]*sizeof(DTYPE), cudaMemcpyDeviceToDevice);
    }
    // First level
    inverse(d_image, d_coeffs[0].data(), d_coeffs[1].data(), d_coeffs[2].data(), d_coeffs[3].data(), tNr[1], tNc[1], Nr, Nc, hlen);

    return 0;
}
    
int nonseparable_wavelet_transform::forward_swt_2D(DTYPE* d_image, wavelet_coeff_t& d_coeffs)
{
    int Nr = winfos.Nr, Nc = winfos.Nc, levels = winfos.nlevels, hlen = winfos.hlen;

    DTYPE* d_tmp1, *d_tmp2;
    d_tmp1 = d_coeffs[0].data();
    d_tmp2 = tmp.data();

    // First level
    forward_swt(d_image, d_coeffs[0].data(), d_coeffs[1].data(), d_coeffs[2].data(), d_coeffs[3].data(), Nr, Nc, hlen, 1);
    for (int i=1; i < levels; i++) {
        forward_swt(d_tmp1, d_tmp2, d_coeffs[3*i+1].data(), d_coeffs[3*i+2].data(), d_coeffs[3*i+3].data(), Nr, Nc, hlen, i+1);
        w_swap_ptr(&d_tmp1, &d_tmp2);
    }
    if ((levels & 1) == 0)
    {
		std::memcpy(d_coeffs[0].data(),tmp.data(), Nr*Nc*sizeof(DTYPE));
    }
    return 0;
}

int nonseparable_wavelet_transform::inverse_swt_2D(DTYPE* d_image, wavelet_coeff_t& d_coeffs)
{
	int Nr = winfos.Nr, Nc = winfos.Nc, levels = winfos.nlevels, hlen = winfos.hlen;
    DTYPE* d_tmp1, *d_tmp2;
    d_tmp1 = d_coeffs[0].data();
    d_tmp2 = tmp.data();

    for (int i = levels-1; i >= 1; i--) {
        inverse_swt(d_tmp2, d_tmp1, d_coeffs[3*i+1].data(), d_coeffs[3*i+2].data(), d_coeffs[3*i+3].data(), Nr, Nc, hlen, i+1);
        w_swap_ptr(&d_tmp1, &d_tmp2);
    }
    if ((levels & 1) == 0)
    {
	   std::memcpy(d_coeffs[0].data(),tmp.data(), Nr*Nc*sizeof(DTYPE));	
	 //	 cudaMemcpy(d_coeffs[0], d_tmp, Nr*Nc*sizeof(DTYPE), cudaMemcpyDeviceToDevice);
    }
    // First scale
    inverse_swt(d_image, d_coeffs[0].data(), d_coeffs[1].data(), d_coeffs[2].data(), d_coeffs[3].data(), Nr, Nc, hlen, 1);
    return 0;
}

void nonseparable_wavelet_transform::forward(DTYPE* img, DTYPE* c_a, DTYPE* c_h, DTYPE* c_v, DTYPE* c_d, int Nr, int Nc, int hlen)
{
	  int Nr_is_odd = (Nr & 1);
    int Nr2 = (Nr + Nr_is_odd)/2;
    int Nc_is_odd = (Nc & 1);
    int Nc2 = (Nc + Nc_is_odd)/2;
	
	 for(int row=0;row<Nr2;++row)
     {
	     for(int col=0;col<Nc2;++col)
	     {
	        int c, hL, hR;
	        if (hlen & 1) { // odd kernel size
	            c = hlen/2;
	            hL = c;
	            hR = c;
	        }
	        else { // even kernel size : center is shifted to the left
	            c = hlen/2 - 1;
	            hL = c;
	            hR = c+1;
	        }
	        DTYPE res_a = 0, res_h = 0, res_v = 0, res_d = 0;
	        DTYPE img_val;
	
	        // Convolution with periodic boundaries extension.
	        // The following can be sped-up by splitting into 3*3 loops, but it would be a nightmare for readability
	        for (int jy = 0; jy <= hR+hL; jy++) {
	            int idx_y = row*2 - c + jy;
	            if (idx_y < 0) idx_y += (Nr + Nr_is_odd); // if N is odd, image is virtually extended
	            // no "else if", since idx_y can be > N-1  after being incremented
	            if (idx_y > Nr-1) {
	                if ((idx_y == Nr) && (Nr_is_odd)) idx_y--; // if N is odd, repeat the right-most element
	                else idx_y -= (Nr + Nr_is_odd); // if N is odd, image is virtually extended
	            }
	
	            for (int jx = 0; jx <= hR+hL; jx++) {
	                int idx_x = col*2 - c + jx;
	                if (idx_x < 0) idx_x += (Nc + Nc_is_odd); // if N is odd, image is virtually extended
	                // no "else if", since idx_x can be > N-1  after being incremented
	                if (idx_x > Nc-1) {
	                    if ((idx_x == Nc) && (Nc_is_odd)) idx_x--; // if N is odd, repeat the right-most element
	                    else idx_x -= (Nc + Nc_is_odd); // if N is odd, image is virtually extended
	                }
	
	                img_val = img[idx_y*Nc + idx_x];
	                res_a += img_val * c_kern_LL[(hlen-1-jy)*hlen + (hlen-1 - jx)];
	                res_h += img_val * c_kern_LH[(hlen-1-jy)*hlen + (hlen-1 - jx)];
	                res_v += img_val * c_kern_HL[(hlen-1-jy)*hlen + (hlen-1 - jx)];
	                res_d += img_val * c_kern_HH[(hlen-1-jy)*hlen + (hlen-1 - jx)];
	            }
	        }
	        c_a[row* Nc2 + col] = res_a;
	        c_h[row* Nc2 + col] = res_h;
	        c_v[row* Nc2 + col] = res_v;
	        c_d[row* Nc2 + col] = res_d;
			 
		 }
	 }
}

void nonseparable_wavelet_transform::inverse(DTYPE* img, DTYPE* c_a, DTYPE* c_h, DTYPE* c_v, DTYPE* c_d, int Nr, int Nc, int Nr2, int Nc2, int hlen)
{
		 for(int row=0;row<Nr2;++row)
     {
	     for(int col=0;col<Nc2;++col)
	     {
			int internal_row = row; 
			int internal_col = col; 
	        int c, hL, hR;
	        int hlen2 = hlen/2; // Convolutions with even/odd indices of the kernels
	        if (hlen2 & 1) { // odd half-kernel size
	            c = hlen2/2;
	            hL = c;
	            hR = c;
	        }
	        else { // even half-kernel size : center is shifted to the RIGHT for reconstruction.
	            c = hlen2/2 - 0;
	            hL = c;
	            hR = c-1;
	            // virtual id for shift
	            // TODO : for the very first convolution (on the edges), this is not exactly accurate (?)
	            internal_col += 1;
	            internal_row += 1;
	        }
	        int jy1 = c - internal_row/2;
	        int jy2 = Nr - 1 - internal_row/2 + c;
	        int jx1 = c - internal_col/2;
	        int jx2 = Nc - 1 - internal_col/2 + c;
	
	        // There are 4 threads/coeff index. Each thread will do a convolution with the even/odd indices of the kernels along each dimension.
	        int offset_x = 1-(internal_col & 1);
	        int offset_y = 1-(internal_row & 1);
	
	        DTYPE res_a = 0, res_h = 0, res_v = 0, res_d = 0;
	        for (int jy = 0; jy <= hR+hL; jy++) {
	            int idx_y = internal_row/2 - c + jy;
	            if (jy < jy1) idx_y += Nr;
	            if (jy > jy2) idx_y -= Nr;
	            for (int jx = 0; jx <= hR+hL; jx++) {
	                int idx_x = internal_col/2 - c + jx;
	                if (jx < jx1) idx_x += Nc;
	                if (jx > jx2) idx_x -= Nc;
	
	                res_a += c_a[idx_y*Nc + idx_x] * c_kern_ILL[(hlen-1- (2*jy + offset_y))*hlen + (hlen-1 - (2*jx + offset_x))];
	                res_h += c_h[idx_y*Nc + idx_x] * c_kern_ILH[(hlen-1- (2*jy + offset_y))*hlen + (hlen-1 - (2*jx + offset_x))];
	                res_v += c_v[idx_y*Nc + idx_x] * c_kern_IHL[(hlen-1- (2*jy + offset_y))*hlen + (hlen-1 - (2*jx + offset_x))];
	                res_d += c_d[idx_y*Nc + idx_x] * c_kern_IHH[(hlen-1- (2*jy + offset_y))*hlen + (hlen-1 - (2*jx + offset_x))];
	            }
	        }
	        if ((hlen2 & 1) == 1) img[internal_row * Nc2 + internal_col] = res_a + res_h + res_v + res_d;
	        else img[(internal_row-1) * Nc2 + (internal_col-1)] = res_a + res_h + res_v + res_d;
	
		 }
	 }
}

void nonseparable_wavelet_transform::forward_swt(DTYPE* img, DTYPE* c_a, DTYPE* c_h, DTYPE* c_v, DTYPE* c_d, int Nr, int Nc, int hlen, int level)
{
	 for(int row=0;row<Nr;++row)
     {
	     for(int col=0;col<Nc;++col)
	     {
	        int factor = 1 << (level - 1);
	        int c, hL, hR;
	        if (hlen & 1) { // odd kernel size
	            c = hlen/2;
	            hL = c;
	            hR = c;
	        }
	        else { // even kernel size : center is shifted to the left
	            c = hlen/2 - 1;
	            hL = c;
	            hR = c+1;
	        }
	
	        c *= factor;
	        int jx1 = c - col;
	        int jx2 = Nc - 1 - col + c;
	        int jy1 = c - row;
	        int jy2 = Nr - 1 - row + c;
	        DTYPE res_a = 0, res_h = 0, res_v = 0, res_d = 0;
	        DTYPE img_val;
	
	        // Convolution with periodic boundaries extension.
	        // The filters are 2-upsampled at each level : [h0, h1, h2, h3] --> [h0, 0, h1, 0, h2, 0, h3, 0]
	        for (int jy = 0; jy <= hR+hL; jy++) {
	            int idx_y = row - c + factor*jy;
	            if (factor*jy < jy1) idx_y += Nr;
	            if (factor*jy > jy2) idx_y -= Nr;
	            for (int jx = 0; jx <= hR+hL; jx++) {
	                int idx_x = col + jx*factor - c;
	                if (factor*jx < jx1) idx_x += Nc;
	                if (factor*jx > jx2) idx_x -= Nc;
	
	                img_val = img[idx_y*Nc + idx_x];
	                res_a += img_val * c_kern_LL[(hlen-1-jy)*hlen + (hlen-1 - jx)];
	                res_h += img_val * c_kern_LH[(hlen-1-jy)*hlen + (hlen-1 - jx)];
	                res_v += img_val * c_kern_HL[(hlen-1-jy)*hlen + (hlen-1 - jx)];
	                res_d += img_val * c_kern_HH[(hlen-1-jy)*hlen + (hlen-1 - jx)];
	
	            }
	        }
	        c_a[row* Nc + col] = res_a;
	        c_h[row* Nc + col] = res_h;
	        c_v[row* Nc + col] = res_v;
	        c_d[row* Nc + col] = res_d;
  
		 }
	 }

}

void nonseparable_wavelet_transform::inverse_swt(DTYPE* img, DTYPE* c_a, DTYPE* c_h, DTYPE* c_v, DTYPE* c_d, int Nr, int Nc, int hlen, int level)
{
		 for(int row=0;row<Nr;++row)
     {
	     for(int col=0;col<Nc;++col)
	     {
	        int factor = 1 << (level - 1);
	        int c, hL, hR;
	        if (hlen & 1) { // odd half-kernel size
	            c = hlen/2;
	            hL = c;
	            hR = c;
	        }
	        else { // even half-kernel size : center is shifted to the RIGHT for reconstruction.
	            c = hlen/2 - 0;
	            hL = c;
	            hR = c-1;
	        }
	        c *= factor;
	        int jy1 = c - row;
	        int jy2 = Nr - 1 - row + c;
	        int jx1 = c - col;
	        int jx2 = Nc - 1 - col + c;
	
	        DTYPE res_a = 0, res_h = 0, res_v = 0, res_d = 0;
	        for (int jy = 0; jy <= hR+hL; jy++) {
	            int idx_y = row - c + jy*factor;
	            if (factor*jy < jy1) idx_y += Nr;
	            if (factor*jy > jy2) idx_y -= Nr;
	            for (int jx = 0; jx <= hR+hL; jx++) {
	                int idx_x = col - c + jx*factor;
	                if (factor*jx < jx1) idx_x += Nc;
	                if (factor*jx > jx2) idx_x -= Nc;
	
	                res_a += c_a[idx_y*Nc + idx_x] * c_kern_ILL[(hlen-1-jy)*hlen + (hlen-1 - jx)]/4;
	                res_h += c_h[idx_y*Nc + idx_x] * c_kern_ILH[(hlen-1-jy)*hlen + (hlen-1 - jx)]/4;
	                res_v += c_v[idx_y*Nc + idx_x] * c_kern_IHL[(hlen-1-jy)*hlen + (hlen-1 - jx)]/4;
	                res_d += c_d[idx_y*Nc + idx_x] * c_kern_IHH[(hlen-1-jy)*hlen + (hlen-1 - jx)]/4;
	            }
	        }
	        img[row * Nc + col] = res_a + res_h + res_v + res_d;
	
		 }
	 }
}

DTYPE* nonseparable_wavelet_transform::w_outer(DTYPE* a, DTYPE* b, int len) {
    DTYPE* res = (DTYPE*) calloc(len*len, sizeof(DTYPE));
    for (int i = 0; i < len; i++) {
        for (int j = 0; j < len; j++) {
            res[i*len+j] = a[i]*b[j];
        }
    }
    return res;
}

