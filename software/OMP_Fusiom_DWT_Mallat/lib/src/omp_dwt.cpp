#include <iostream>
#include <vector>
#include <array>
#include <cmath>
#include "omp_dwt.h"
#include "wt.h"
#include "filters.h"

void add_cdf_53_wavelet(char* filtername,Wavelets& W)
{
    // Example of custom filter (here: LeGall 5/3 wavelet --> see "Document and Image compression", Barni, 2006)
    std::array<DTYPE,6> filter1;
    std::array<DTYPE,6> filter2;
    std::array<DTYPE,6> ifilter1;
    std::array<DTYPE,6> ifilter2;
   

    filter1[0] = 0.0;
    filter1[1] = -1.0/8;
    filter1[2] = 2.0/8;
    filter1[3] = 6.0/8;
    filter1[4] = 2.0/8;
    filter1[5] = -1.0/8;

    filter2[0] = 0;
    filter2[1] = -0.5;
    filter2[2] = 1.0;
    filter2[3] = -0.5;
    filter2[4] = 0.0;
    filter2[5] = 0;

    ifilter1[0] = 0;
    ifilter1[1] = 0.5;
    ifilter1[2] = 1;
    ifilter1[3] = 0.5;
    ifilter1[4] = 0;
    ifilter1[5] = 0;

    ifilter2[0] = 0;
    ifilter2[1] = -1.0/8;
    ifilter2[2] = -2.0/8;
    ifilter2[3] = 6.0/8;
    ifilter2[4] = -2.0/8;
    ifilter2[5] = -1.0/8;

    W.set_filters_forward(filtername, filter1.size(), filter1.data(), filter2.data());
    W.set_filters_inverse(ifilter1.data(), ifilter2.data());
}

void add_cdf_97_wavelet(char* filtername,Wavelets& W)
{
    // Example of custom filter (here: CDF 9/7 wavelet)
    std::array<DTYPE,10> filter1;
    std::array<DTYPE,10> filter2;
    std::array<DTYPE,10> ifilter1;
    std::array<DTYPE,10> ifilter2;

    filter1[0] = 0.0;
    filter1[1] = 0.026748757411 ;
    filter1[2] = -0.016864118443;
    filter1[3] = -0.078223266529;
    filter1[4] = 0.266864118443 ;
    filter1[5] = 0.602949018236 ;
    filter1[6] = 0.266864118443 ;
    filter1[7] = -0.078223266529;
    filter1[8] = -0.016864118443;
    filter1[9] = 0.026748757411 ;

    filter2[0] = 0              ;
    filter2[1] = 0.091271763114 ;
    filter2[2] = -0.057543526229;
    filter2[3] = -0.591271763114;
    filter2[4] =    1.11508705  ;
    filter2[5] = -0.591271763114;
    filter2[6] = -0.057543526229;
    filter2[7] = 0.091271763114 ;
    filter2[8] = 0              ;
    filter2[9] = 0              ;

    ifilter1[0] = 0               ;
    ifilter1[1] = -0.091271763114 ;
    ifilter1[2] = -0.057543526229 ;
    ifilter1[3] = 0.591271763114  ;
    ifilter1[4] = 1.11508705      ;
    ifilter1[5] = 0.591271763114  ;
    ifilter1[6] = -0.057543526229 ;
    ifilter1[7] = -0.091271763114 ;
    ifilter1[8] = 0               ;
    ifilter1[9] = 0               ;

    ifilter2[0] = 0.0;
    ifilter2[1] = 0.026748757411 ;
    ifilter2[2] = 0.016864118443 ;
    ifilter2[3] = -0.078223266529;
    ifilter2[4] = -0.266864118443;
    ifilter2[5] = 0.602949018236 ;
    ifilter2[6] = -0.266864118443;
    ifilter2[7] = -0.078223266529;
    ifilter2[8] = 0.016864118443 ;
    ifilter2[9] = 0.026748757411 ;

    W.set_filters_forward(filtername, filter1.size(), filter1.data(), filter2.data());
    W.set_filters_inverse(ifilter1.data(), ifilter2.data());   
}


void calculate_dwt(int Nr,int Nc)
{
        std::vector<DTYPE> img_in(Nr*Nc,0);

	    for(int y=0;y<Nr;y++)
	        for(int x=0;x<Nc;x++)
	                img_in[y*Nc+x] = (DTYPE)x+y;
	    
	    int nlevels, do_separable = 1, do_swt = 0;
	    int do_cycle_spinning = 0;
	    nlevels = 5;            
		
		
	    char* filtername = (char*) "LeGall 5/3";
	    Wavelets W(img_in.data(), Nr, Nc, filtername, nlevels, 1, do_separable, do_cycle_spinning, do_swt);    
	    add_cdf_53_wavelet(filtername,W);
	
	    W.print_informations();	
		
		W.forward_2D_even_symmetric_cdf_53_wavelet();
	    W.inverse_2D_even_symmetric_cdf_53_wavelet();
	
	    std::vector<DTYPE> img_out(Nr*Nc,0);
	    W.get_image(img_out.data());
	 
	    int n_errors = 0;
	   
	    for(int y=0;y<Nr;y++)
	    {
	      for(int x=0;x<Nc;x++)
	      {
	         auto in_pix = img_in[y*Nc+x];
	         auto rec_pix = img_out[y*Nc+x];
	         auto diff = std::fabs(in_pix-rec_pix);
	
	        if(diff>1e-3) 
	        {
			//	std::cout<<"Found difference at (x,y):("<<x<<","<<y<<")"<<std::endl;
			    ++n_errors;
			}
	      }
	    }
		std::cout << "Errors 5/3: " << n_errors << std::endl;
	        
	    char* filtername_97 = (char*) "CDF 9/7";    
	   // Wavelets W(img_float_in.data(), Nr, Nc, filtername, nlevels, 1, do_separable, do_cycle_spinning, do_swt);       
	    add_cdf_97_wavelet(filtername_97,W);
	
	    //nlevels = W.winfos.nlevels;
	    W.print_informations();	
	
		W.forward_2D_even_symmetric_cdf_97_wavelet();
	    W.inverse_2D_even_symmetric_cdf_97_wavelet();
	    W.get_image(img_out.data());
	
		
	    n_errors = 0;
	   
	    for(int y=0;y<Nr;y++)
	    {
	      for(int x=0;x<Nc;x++)
	      {
	         auto in_pix = img_in[y*Nc+x];
	         auto rec_pix = img_out[y*Nc+x];
	         auto diff = std::fabs(in_pix-rec_pix);
	
	        if(diff>1e-3) 
	        {
			//	std::cout<<"Found difference at (x,y):("<<x<<","<<y<<")"<<std::endl;
			    ++n_errors;
			}
	      }
	    }
		std::cout << "Errors 9/7: " << n_errors << std::endl;
       

}

int main()
{

/*	int Nr_base = 512;
	int Nc_base = 512;
	
	for(int scale = 0;scale < 4;++scale) 
	{
        const int Nr = Nr_base*std::pow(2,scale);      
        const int Nc = Nc_base*std::pow(2,scale);
        calculate_dwt(Nr,Nc);      
 	}*/

    calculate_dwt(512,512);      
    calculate_dwt(768,576);      
    calculate_dwt(1024,1024);      
    calculate_dwt(2048,1024);      
    calculate_dwt(2048,2048);      
    calculate_dwt(4096,4096);      
	
	return 0; 
}

