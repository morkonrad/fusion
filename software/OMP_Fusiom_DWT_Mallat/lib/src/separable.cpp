#include "separable.h"
#include "common.h"
#include <iostream>
#include <cstring>
#include <string>
#include <array>
#include <cmath>

#include <functional>
#include <chrono>
#include <algorithm>

#define TIMING

#ifdef _OPENMP
 # include <omp.h>

struct measure_omp_wtime_t{
   measure_omp_wtime_t(const char* msg):message(msg)
   {
     start_time = omp_get_wtime(); 
   }
   
   void stop()
   {
	  double end_time = omp_get_wtime();
      std::cout << message << " used time: " << (end_time - start_time) << std::endl;
   }
   double start_time;
   
   std::string message;
};



#else //_OPENMP


struct measure_omp_wtime_t{
   measure_omp_wtime_t(const char* msg):message(msg)
   {
     start_time =  std::chrono::high_resolution_clock::now(); 
   }
   
   void stop()
   {
	  auto end_time =  std::chrono::high_resolution_clock::now() ;
	  auto duration = std::chrono::duration_cast<std::chrono::duration<double>>( (end_time - start_time));
      std::cout << message << " used time: " << duration.count() << std::endl;
   }
   std::chrono::high_resolution_clock::time_point start_time;
   
   std::string message;
};



#endif //_OPENMP


#define START_TIMING_MEASUREMENT(msg)\
 measure_omp_wtime_t measure_omp_wtime(msg);\
  do{}while(false)


#define STOP_TIMING_MEASUREMENT\
  measure_omp_wtime.stop();\
  do{}while(false)


namespace{


template <
		typename TimeT = std::chrono::milliseconds, class ClockT = std::chrono::system_clock
	> 
	struct measure
	{
		/**
		* @ fn    execution
		* @ brief Returns the quantity (count) of the elapsed time as TimeT units
		*/
		using time_base = TimeT;
		
		template<typename F, typename ...Args>
		static typename TimeT::rep execution(F&& func, Args&&... args)
		{
			auto start = ClockT::now();
			
			std::invoke(std::forward<decltype(func)>(func), std::forward<Args>(args)...);
			//std::forward<decltype(func)>(func)(std::forward<Args>(args)...);

			auto duration = std::chrono::duration_cast<TimeT>(ClockT::now() - start);

			return duration.count();
		}

		/**
		* @ fn    duration
		* @ brief Returns the duration (in chrono's type system) of the elapsed time
		*/
		template<typename F, typename... Args>
		static TimeT duration(F&& func, Args&&... args)
		{
			auto start = ClockT::now();
			
			//std::forward<decltype(func)>(func)(std::forward<Args>(args)...);
			std::invoke(std::forward<decltype(func)>(func), std::forward<Args>(args)...);
			
			return std::chrono::duration_cast<TimeT>(ClockT::now() - start);
		}
};

  using measure_type = measure<std::chrono::nanoseconds>;
  const char* measure_result_unit = " ns";

void create_output(const char* method,int level,double call_duration,double variance,int Nrow_old,int Ncol_old,int Nrow_new,int Ncol_new)
{
	const char* separ = " : ";
	std::cout << method << separ
	          << "level " << level << separ
	          << call_duration << measure_result_unit << separ
	          << std::sqrt(variance) << measure_result_unit << separ
	          << Nrow_old <<"x" << Ncol_old << "->" << Nrow_new <<"x" << Ncol_new <<  std::endl;
}  

  
}




separable_wavelet_transform::separable_wavelet_transform(w_info winfo):winfos(winfo)
{
    tmp.resize(2*winfos.Nr*winfos.Nc);
    //d_tmp = tmp.data();
}

separable_wavelet_transform::~separable_wavelet_transform()
{
}
	
int separable_wavelet_transform::set_filters(const char* wname, int do_swt)
{
    int hlen = 0;
    DTYPE* f1_l, *f1_h, *f1_il, *f1_ih;

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
            f1_l = all_filters[i].f_l;
            f1_h = all_filters[i].f_h;
            f1_il = all_filters[i].i_l;
            f1_ih = all_filters[i].i_h;
            break;
        }
    }
    if (hlen == 0) {
        printf("ERROR: w_compute_filters(): unknown filter %s\n", wname);
        return -2;
    }

    // Copy the filters to device constant memory
   
    c_kern_L.resize(hlen);
    c_kern_H.resize(hlen);
    c_kern_IL.resize(hlen);
    c_kern_IH.resize(hlen);
    
    std::memcpy(c_kern_L.data(),f1_l, hlen*sizeof(DTYPE));
    std::memcpy(c_kern_H.data(),f1_h, hlen*sizeof(DTYPE));
    std::memcpy(c_kern_IL.data(),f1_il, hlen*sizeof(DTYPE));
    std::memcpy(c_kern_IH.data(),f1_ih, hlen*sizeof(DTYPE));
    
    winfos.hlen = hlen;
    return hlen;
	
}

int separable_wavelet_transform::set_filters_forward(DTYPE* filter1, DTYPE* filter2, uint len)
{


    c_kern_L.resize(len);
    c_kern_H.resize(len);
	
    std::memcpy(c_kern_L.data(), filter1, len*sizeof(DTYPE));
    std::memcpy(c_kern_H.data(), filter2, len*sizeof(DTYPE));
    winfos.hlen = len;
 
    return 0;
}
    
    
int separable_wavelet_transform::set_filters_inverse(DTYPE* filter1, DTYPE* filter2, uint len)
{

    c_kern_IL.resize(len);
    c_kern_IH.resize(len);

    std::memcpy(c_kern_IL.data(), filter1, len*sizeof(DTYPE));
    std::memcpy(c_kern_IH.data(), filter2, len*sizeof(DTYPE));
    winfos.hlen = len;

    return 0;
}
   
int separable_wavelet_transform::forward_2D(DTYPE* d_image, wavelet_coeff_t& d_coeffs)
{
	int Nr = winfos.Nr, Nc = winfos.Nc, levels = winfos.nlevels, hlen = winfos.hlen;
    int Nc2 = Nc, Nr2 = Nr;
    int Nc2_old = Nc2, Nr2_old = Nr;
    w_div2(&Nc2);
    w_div2(&Nr2);
    // d_tmp can have up to 2*Nr*Nc elemets (two input images) [Nr*Nc would be enough here].
    // Here d_tmp1 (resp. d_tmp2) is used for the horizontal (resp. vertical) downsampling.
    // Given a dimension size N, the subsampled dimension size is N/2 if N is even, (N+1)/2 otherwise.
    DTYPE* d_tmp1 = tmp.data();
    DTYPE* d_tmp2 = tmp.data()+ Nr*Nc2;

    // First level

    START_TIMING_MEASUREMENT("Separable Forward");
    
    
    forward_pass1(d_image, d_tmp1, d_tmp2, Nr2_old, Nc2_old, hlen);
    forward_pass2(d_tmp1, d_tmp2, d_coeffs[0].data(), d_coeffs[1].data(), d_coeffs[2].data(), d_coeffs[3].data(), Nr2_old, Nc2, hlen);
	 
    for (int i=1; i < levels; i++) {
        Nc2_old = Nc2; Nr2_old = Nr2;
        w_div2(&Nc2);
        w_div2(&Nr2);
        forward_pass1(d_coeffs[0].data(), d_tmp1, d_tmp2, Nr2_old, Nc2_old, hlen);
        forward_pass2(d_tmp1, d_tmp2, d_coeffs[0].data(), d_coeffs[3*i+1].data(), d_coeffs[3*i+2].data(), d_coeffs[3*i+3].data(), Nr2_old, Nc2, hlen);
    }
    STOP_TIMING_MEASUREMENT;
  
    return 0;
}

int separable_wavelet_transform::forward_2D_even_symmetric(DTYPE* d_image, wavelet_coeff_t& d_coeffs)
{
	int Nr = winfos.Nr, Nc = winfos.Nc, levels = winfos.nlevels, hlen = winfos.hlen;
    int Nc2 = Nc, Nr2 = Nr;
    int Nc2_old = Nc2, Nr2_old = Nr;
    w_div2(&Nc2);
    w_div2(&Nr2);
    // d_tmp can have up to 2*Nr*Nc elemets (two input images) [Nr*Nc would be enough here].
    // Here d_tmp1 (resp. d_tmp2) is used for the horizontal (resp. vertical) downsampling.
    // Given a dimension size N, the subsampled dimension size is N/2 if N is even, (N+1)/2 otherwise.
    DTYPE* d_tmp1 = tmp.data();
    DTYPE* d_tmp2 = tmp.data()+ Nr*Nc2;

    // First level
    START_TIMING_MEASUREMENT("Separable Forward even symmetric");

    forward_pass1_even_symmetric_unroll4(d_image, d_tmp1, d_tmp2, Nr2_old, Nc2_old, hlen);
    forward_pass2_even_symmetric_unroll4(d_tmp1, d_tmp2, d_coeffs[0].data(), d_coeffs[1].data(), d_coeffs[2].data(), d_coeffs[3].data(), Nr2_old, Nc2, hlen);
	 
    for (int i=1; i < levels; i++) {
        Nc2_old = Nc2; Nr2_old = Nr2;
        w_div2(&Nc2);
        w_div2(&Nr2);
        forward_pass1_even_symmetric_unroll4(d_coeffs[0].data(), d_tmp1, d_tmp2, Nr2_old, Nc2_old, hlen);
        forward_pass2_even_symmetric_unroll4(d_tmp1, d_tmp2, d_coeffs[0].data(), d_coeffs[3*i+1].data(), d_coeffs[3*i+2].data(), d_coeffs[3*i+3].data(), Nr2_old, Nc2, hlen);
    }

    STOP_TIMING_MEASUREMENT;

    return 0;
}

int separable_wavelet_transform::forward_2D_even_symmetric_cdf_97_wavelet(DTYPE* d_image, wavelet_coeff_t& d_coeffs)
{
	int Nr = winfos.Nr, Nc = winfos.Nc, levels = winfos.nlevels, hlen = winfos.hlen;
    int Nc2 = Nc, Nr2 = Nr;
    int Nc2_old = Nc2, Nr2_old = Nr;
    w_div2(&Nc2);
    w_div2(&Nr2);
    // d_tmp can have up to 2*Nr*Nc elemets (two input images) [Nr*Nc would be enough here].
    // Here d_tmp1 (resp. d_tmp2) is used for the horizontal (resp. vertical) downsampling.
    // Given a dimension size N, the subsampled dimension size is N/2 if N is even, (N+1)/2 otherwise.
    DTYPE* d_tmp1 = tmp.data();
    DTYPE* d_tmp2 = tmp.data()+ Nr*Nc2;

    // First level
 //   measure_type = measure<std::chrono::nanoseconds>;
 // const char* measure_result_unit = "ns";
#ifdef TIMING
  
    double avg_pass1{0};
    double avg_pass2{0};
    double var_pass1{0};
    double var_pass2{0};
    double K1,K2;
    
    for(int i=0;i<100;++i)
    {
     DTYPE* d_tmp1 = tmp.data();
     DTYPE* d_tmp2 = tmp.data()+ Nr*Nc2;
     auto t1 = measure_type::duration(&separable_wavelet_transform::forward_pass1_even_symmetric_cdf_97_wavelet,this,d_image, d_tmp1, d_tmp2, Nr2_old, Nc2_old, hlen);
     auto t2 = measure_type::duration(&separable_wavelet_transform::forward_pass2_even_symmetric_cdf_97_wavelet,this,d_tmp1, d_tmp2, d_coeffs[0].data(), d_coeffs[1].data(), d_coeffs[2].data(), d_coeffs[3].data(), Nr2_old, Nc2, hlen);
     if(i==0)
     {
       K1 = t1.count();
       K2 = t2.count();
     }
     else
     {
		avg_pass1 += (t1.count()-K1); 
		avg_pass2 += (t2.count()-K2); 
		var_pass1 += (t1.count()-K1)*(t1.count()-K1); 
		var_pass2 += (t2.count()-K2)*(t2.count()-K2); 
	 }
	 }
	 
	 double var1 = (var_pass1-(avg_pass1*avg_pass1)/double(100))/double(100-1);
	 double var2 = (var_pass2-(avg_pass2*avg_pass2)/double(100))/double(100-1);

	 double avg1 = K1 + avg_pass1/double(100);
	 double avg2 = K2 + avg_pass2/double(100);
	 
	 create_output("forward_pass1_even_symmetric_cdf_97_wavelet",0,avg1,var1,Nr2_old,Nc2_old,Nr2_old,Nc2);
     create_output("forward_pass2_even_symmetric_cdf_97_wavelet",0,avg2,var2,Nr2_old,Nc2,Nc2,Nr2);

     DTYPE* d_tmp1_pass1 = d_tmp1;
     DTYPE* d_tmp2_pass1 = d_tmp2;

     for(int i=1; i < levels; i++) {
        double  avg_pass1{0};
        double avg_pass2{0};
        double var_pass1{0};
        double var_pass2{0};
        double K1,K2;

        Nc2_old = Nc2; Nr2_old = Nr2;
        w_div2(&Nc2);
        w_div2(&Nr2);
        aligned_data_vec_t temp_coeffs0( d_coeffs[0].begin(), d_coeffs[0].end());
        for(int j=0;j<100;++j)
        {
		   d_tmp1_pass1 = d_tmp1;
           d_tmp2_pass1 = d_tmp2;
           std::copy( d_coeffs[0].begin(), d_coeffs[0].end(),temp_coeffs0.begin());
           auto t1 = measure_type::duration(&separable_wavelet_transform::forward_pass1_even_symmetric_cdf_97_wavelet,this,temp_coeffs0.data(), d_tmp1_pass1, d_tmp2_pass1, Nr2_old, Nc2_old, hlen);
           auto t2 = measure_type::duration(&separable_wavelet_transform::forward_pass2_even_symmetric_cdf_97_wavelet,this,d_tmp1_pass1, d_tmp2_pass1, temp_coeffs0.data(), d_coeffs[3*i+1].data(), d_coeffs[3*i+2].data(), d_coeffs[3*i+3].data(), Nr2_old, Nc2, hlen);
           if(i==0)
           {
            K1 = t1.count();
            K2 = t2.count();
           }
           else
           {
		     avg_pass1 += (t1.count()-K1); 
		     avg_pass2 += (t2.count()-K2); 
		     var_pass1 += (t1.count()-K1)*(t1.count()-K1); 
		     var_pass2 += (t2.count()-K2)*(t2.count()-K2); 
	       }
        }  
        d_tmp1 = d_tmp1_pass1;
        d_tmp2 = d_tmp2_pass1;
        std::copy(temp_coeffs0.begin(),temp_coeffs0.end(),d_coeffs[0].begin());
   	    double var1 = (var_pass1-(avg_pass1*avg_pass1)/double(100))/double(100-1);
	    double var2 = (var_pass2-(avg_pass2*avg_pass2)/double(100))/double(100-1);

	    double avg1 = K1 + avg_pass1/double(100);
	    double avg2 = K2 + avg_pass2/double(100);
        
        create_output("forward_pass1_even_symmetric_cdf_97_wavelet",i,avg1,var1,Nr2_old,Nc2_old,Nr2_old,Nc2);
        create_output("forward_pass2_even_symmetric_cdf_97_wavelet",i,avg2,var2,Nr2_old,Nc2,Nc2,Nr2);
        
    }




#else

    START_TIMING_MEASUREMENT("Separable Forward even symmetric cdf_97");

    
    forward_pass1_even_symmetric_cdf_97_wavelet(d_image, d_tmp1, d_tmp2, Nr2_old, Nc2_old, hlen);
    forward_pass2_even_symmetric_cdf_97_wavelet(d_tmp1, d_tmp2, d_coeffs[0].data(), d_coeffs[1].data(), d_coeffs[2].data(), d_coeffs[3].data(), Nr2_old, Nc2, hlen);

	 
    for (int i=1; i < levels; i++) {
        Nc2_old = Nc2; Nr2_old = Nr2;
        w_div2(&Nc2);
        w_div2(&Nr2);
        forward_pass1_even_symmetric_cdf_97_wavelet(d_coeffs[0].data(), d_tmp1, d_tmp2, Nr2_old, Nc2_old, hlen);
        forward_pass2_even_symmetric_cdf_97_wavelet(d_tmp1, d_tmp2, d_coeffs[0].data(), d_coeffs[3*i+1].data(), d_coeffs[3*i+2].data(), d_coeffs[3*i+3].data(), Nr2_old, Nc2, hlen);
    }

    STOP_TIMING_MEASUREMENT;
#endif

    return 0;
}

int separable_wavelet_transform::forward_2D_even_symmetric_cdf_53_wavelet(DTYPE* d_image, wavelet_coeff_t& d_coeffs)
{
	int Nr = winfos.Nr, Nc = winfos.Nc, levels = winfos.nlevels, hlen = winfos.hlen;
    int Nc2 = Nc, Nr2 = Nr;
    int Nc2_old = Nc2, Nr2_old = Nr;
    w_div2(&Nc2);
    w_div2(&Nr2);
    // d_tmp can have up to 2*Nr*Nc elemets (two input images) [Nr*Nc would be enough here].
    // Here d_tmp1 (resp. d_tmp2) is used for the horizontal (resp. vertical) downsampling.
    // Given a dimension size N, the subsampled dimension size is N/2 if N is even, (N+1)/2 otherwise.
    DTYPE* d_tmp1 = tmp.data();
    DTYPE* d_tmp2 = tmp.data()+ Nr*Nc2;

    // First level
#ifdef TIMING
  
    double avg_pass1{0};
    double avg_pass2{0};
    double var_pass1{0};
    double var_pass2{0};
    double K1,K2;
  
    for(int i=0;i<100;++i)
    {
     DTYPE* d_tmp1 = tmp.data();
     DTYPE* d_tmp2 = tmp.data()+ Nr*Nc2;
     auto t1 = measure_type::duration(&separable_wavelet_transform::forward_pass1_even_symmetric_cdf_53_wavelet,this,d_image, d_tmp1, d_tmp2, Nr2_old, Nc2_old, hlen);
     auto t2 = measure_type::duration(&separable_wavelet_transform::forward_pass2_even_symmetric_cdf_53_wavelet,this,d_tmp1, d_tmp2, d_coeffs[0].data(), d_coeffs[1].data(), d_coeffs[2].data(), d_coeffs[3].data(), Nr2_old, Nc2, hlen);
     if(i==0)
     {
       K1 = t1.count();
       K2 = t2.count();
     }
     else
     {
		avg_pass1 += (t1.count()-K1); 
		avg_pass2 += (t2.count()-K2); 
		var_pass1 += (t1.count()-K1)*(t1.count()-K1); 
		var_pass2 += (t2.count()-K2)*(t2.count()-K2); 
	 }
	}
	 double var1 = (var_pass1-(avg_pass1*avg_pass1)/double(100))/double(100-1);
	 double var2 = (var_pass2-(avg_pass2*avg_pass2)/double(100))/double(100-1);

	 double avg1 = K1 + avg_pass1/double(100);
	 double avg2 = K2 + avg_pass2/double(100);
	
     create_output("forward_pass1_even_symmetric_cdf_53_wavelet",0,avg1,var1,Nr2_old,Nc2_old,Nr2_old,Nc2);
     create_output("forward_pass2_even_symmetric_cdf_53_wavelet",0,avg2,var2,Nr2_old,Nc2,Nc2,Nr2);
	 

     DTYPE* d_tmp1_pass1 = d_tmp1;
     DTYPE* d_tmp2_pass1 = d_tmp2;

    for (int i=1; i < levels; i++) {
        double  avg_pass1{0};
        double avg_pass2{0};
        double var_pass1{0};
        double var_pass2{0};
        double K1,K2;

        Nc2_old = Nc2; Nr2_old = Nr2;
        w_div2(&Nc2);
        w_div2(&Nr2);
        aligned_data_vec_t temp_coeffs0( d_coeffs[0].begin(), d_coeffs[0].end());
        for(int j=0;j<100;++j)
        {
		   d_tmp1_pass1 = d_tmp1;
           d_tmp2_pass1 = d_tmp2;
           std::copy( d_coeffs[0].begin(), d_coeffs[0].end(),temp_coeffs0.begin());
           auto t1 = measure_type::duration(&separable_wavelet_transform::forward_pass1_even_symmetric_cdf_53_wavelet,this,temp_coeffs0.data(), d_tmp1_pass1, d_tmp2_pass1, Nr2_old, Nc2_old, hlen);
           auto t2 = measure_type::duration(&separable_wavelet_transform::forward_pass2_even_symmetric_cdf_53_wavelet,this,d_tmp1_pass1, d_tmp2_pass1, temp_coeffs0.data(), d_coeffs[3*i+1].data(), d_coeffs[3*i+2].data(), d_coeffs[3*i+3].data(), Nr2_old, Nc2, hlen);
           if(i==0)
           {
            K1 = t1.count();
            K2 = t2.count();
           }
           else
           {
		     avg_pass1 += (t1.count()-K1); 
		     avg_pass2 += (t2.count()-K2); 
		     var_pass1 += (t1.count()-K1)*(t1.count()-K1); 
		     var_pass2 += (t2.count()-K2)*(t2.count()-K2); 
	       }
        }  
        d_tmp1 = d_tmp1_pass1;
        d_tmp2 = d_tmp2_pass1;
        std::copy(temp_coeffs0.begin(),temp_coeffs0.end(),d_coeffs[0].begin());

   	    double var1 = (var_pass1-(avg_pass1*avg_pass1)/double(100))/double(100-1);
	    double var2 = (var_pass2-(avg_pass2*avg_pass2)/double(100))/double(100-1);

	    double avg1 = K1 + avg_pass1/double(100);
	    double avg2 = K2 + avg_pass2/double(100);
        

        create_output("forward_pass1_even_symmetric_cdf_53_wavelet",i,avg1,var1,Nr2_old,Nc2_old,Nr2_old,Nc2);
        create_output("forward_pass2_even_symmetric_cdf_53_wavelet",i,avg2,var2,Nr2_old,Nc2,Nc2,Nr2);
        
    }




#else

    START_TIMING_MEASUREMENT("Separable Forward even symmetric cdf_53");

    
    forward_pass1_even_symmetric_cdf_53_wavelet(d_image, d_tmp1, d_tmp2, Nr2_old, Nc2_old, hlen);
    forward_pass2_even_symmetric_cdf_53_wavelet(d_tmp1, d_tmp2, d_coeffs[0].data(), d_coeffs[1].data(), d_coeffs[2].data(), d_coeffs[3].data(), Nr2_old, Nc2, hlen);

	 
    for (int i=1; i < levels; i++) {
        Nc2_old = Nc2; Nr2_old = Nr2;
        w_div2(&Nc2);
        w_div2(&Nr2);
        forward_pass1_even_symmetric_cdf_53_wavelet(d_coeffs[0].data(), d_tmp1, d_tmp2, Nr2_old, Nc2_old, hlen);
        forward_pass2_even_symmetric_cdf_53_wavelet(d_tmp1, d_tmp2, d_coeffs[0].data(), d_coeffs[3*i+1].data(), d_coeffs[3*i+2].data(), d_coeffs[3*i+3].data(), Nr2_old, Nc2, hlen);
    }

    STOP_TIMING_MEASUREMENT;
#endif

    return 0;
}



int separable_wavelet_transform::forward_1D(DTYPE* d_image,wavelet_coeff_t& d_coeffs)
{
	int Nr = winfos.Nr, Nc = winfos.Nc, levels = winfos.nlevels, hlen = winfos.hlen;
    DTYPE* d_tmp1 = d_coeffs[0].data();
    DTYPE* d_tmp2 = tmp.data();
    // First level
    int Nc2 = Nc;
    int Nc2_old = Nc2;
    w_div2(&Nc2);
    forward_pass1(d_image, d_coeffs[0].data(), d_coeffs[1].data(), Nr, Nc, hlen);
    for (int i=1; i < levels; i++) {
        Nc2_old = Nc2;
        w_div2(&Nc2);
        forward_pass1(d_tmp1, d_tmp2, d_coeffs[i+1].data(), Nr, Nc2_old, hlen);
        w_swap_ptr(&d_tmp1, &d_tmp2);
    }
    if ((levels > 1) && ((levels & 1) == 0))
    {
		 std::memcpy(d_coeffs[0].data(), tmp.data(), Nr*Nc2*sizeof(DTYPE));
		 
    }
    return 0;
}

int separable_wavelet_transform::inverse_2D(DTYPE* d_image, wavelet_coeff_t& d_coeffs)
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
    DTYPE* d_tmp1 = tmp.data();
    DTYPE* d_tmp2 = tmp.data() + Nr*tNc[1];

    // TODO: variables for better readability instead of tNr[i], tNc[i]
   START_TIMING_MEASUREMENT("Separable Inverse");
   

    for (int i = levels-1; i >= 1; i--) {
        inverse_pass1(d_coeffs[0].data(), d_coeffs[3*i+1].data(), d_coeffs[3*i+2].data(), d_coeffs[3*i+3].data(), d_tmp1, d_tmp2, tNr[i+1], tNc[i+1], tNr[i], hlen);
        inverse_pass2(d_tmp1, d_tmp2, d_coeffs[0].data(), tNr[i], tNc[i+1], tNc[i], hlen);


    }
    // First scale
    inverse_pass1(d_coeffs[0].data(), d_coeffs[1].data(), d_coeffs[2].data(), d_coeffs[3].data(), d_tmp1, d_tmp2, tNr[1], tNc[1], tNr[0], hlen);
    inverse_pass2(d_tmp1, d_tmp2, d_image, tNr[0], tNc[1], tNc[0], hlen);

    STOP_TIMING_MEASUREMENT;

    return 0;
}

int separable_wavelet_transform::inverse_2D_even_symmetric_cdf_97_wavelet(DTYPE* d_image, wavelet_coeff_t& d_coeffs)
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
    DTYPE* d_tmp1 = tmp.data();
    DTYPE* d_tmp2 = tmp.data() + Nr*tNc[1];


#ifdef TIMING
  
     DTYPE* d_tmp1_pass1 = d_tmp1;
     DTYPE* d_tmp2_pass1 = d_tmp2;
  
     for (int i = levels-1; i >= 1; i--) {
        double avg_pass1{0};
        double avg_pass2{0};
        double var_pass1{0};
        double var_pass2{0};
        double K1,K2;
        aligned_data_vec_t temp_coeffs0( d_coeffs[0].begin(), d_coeffs[0].end());    
        for(int j=0;j<100;++j)
        {   
  	      d_tmp1_pass1 = d_tmp1;
          d_tmp2_pass1 = d_tmp2;
          std::copy( d_coeffs[0].begin(), d_coeffs[0].end(),temp_coeffs0.begin());
          auto t1 = measure_type::duration(&separable_wavelet_transform::inverse_pass1_even_symmetric_cdf_97_wavelet,this,temp_coeffs0.data(), d_coeffs[3*i+1].data(), d_coeffs[3*i+2].data(), d_coeffs[3*i+3].data(), d_tmp1_pass1, d_tmp2_pass1, tNr[i+1], tNc[i+1], tNr[i], hlen);
          auto t2 = measure_type::duration(&separable_wavelet_transform::inverse_pass2_even_symmetric_cdf_97_wavelet,this,d_tmp1_pass1, d_tmp2_pass1, temp_coeffs0.data(), tNr[i], tNc[i+1], tNc[i], hlen);
           if(j==0)
           {
            K1 = t1.count();
            K2 = t2.count();
           }
           else
           {
		     avg_pass1 += (t1.count()-K1); 
		     avg_pass2 += (t2.count()-K2); 
		     var_pass1 += (t1.count()-K1)*(t1.count()-K1); 
		     var_pass2 += (t2.count()-K2)*(t2.count()-K2); 
	       }

        }  
        d_tmp1 = d_tmp1_pass1;
        d_tmp2 = d_tmp2_pass1;
        std::copy(temp_coeffs0.begin(),temp_coeffs0.end(),d_coeffs[0].begin());
     	double var1 = (var_pass1-(avg_pass1*avg_pass1)/double(100))/double(100-1);
	    double var2 = (var_pass2-(avg_pass2*avg_pass2)/double(100))/double(100-1);

	    double avg1 = K1 + avg_pass1/double(100);
	    double avg2 = K2 + avg_pass2/double(100);

       // d_coeffs0_pass1 = d_coeffs[0].data();
        create_output("inverse_pass1_even_symmetric_cdf_97_wavelet",i,avg1,var1, tNr[i+1], tNc[i+1],tNr[i],tNc[i+1]);
        create_output("inverse_pass2_even_symmetric_cdf_97_wavelet",i,avg2,var2,tNr[i],tNc[i+1],tNr[i],tNc[i]);
   }
    
    // First scale
    double avg_pass1{0};
    double avg_pass2{0};
    double var_pass1{0};
    double var_pass2{0};
    double K1,K2;

    for(int i=0;i<100;++i)
    {
      auto t1 = measure_type::duration(&separable_wavelet_transform::inverse_pass1_even_symmetric_cdf_97_wavelet,this,d_coeffs[0].data(), d_coeffs[1].data(), d_coeffs[2].data(), d_coeffs[3].data(), d_tmp1, d_tmp2, tNr[1], tNc[1], tNr[0], hlen);
      auto t2 = measure_type::duration(&separable_wavelet_transform::inverse_pass2_even_symmetric_cdf_97_wavelet,this,d_tmp1, d_tmp2, d_image, tNr[0], tNc[1], tNc[0], hlen);
     if(i==0)
     {
       K1 = t1.count();
       K2 = t2.count();
     }
     else
     {
		avg_pass1 += (t1.count()-K1); 
		avg_pass2 += (t2.count()-K2); 
		var_pass1 += (t1.count()-K1)*(t1.count()-K1); 
		var_pass2 += (t2.count()-K2)*(t2.count()-K2); 
	 }

    } 
	
     double var1 = (var_pass1-(avg_pass1*avg_pass1)/double(100))/double(100-1);
	 double var2 = (var_pass2-(avg_pass2*avg_pass2)/double(100))/double(100-1);

	 double avg1 = K1 + avg_pass1/double(100);
	 double avg2 = K2 + avg_pass2/double(100);

    create_output("inverse_pass1_even_symmetric_cdf_97_wavelet",0,avg1,var1,tNr[1], tNc[1], tNr[0],tNc[1]);
    create_output("inverse_pass2_even_symmetric_cdf_97_wavelet",0,avg2,var2,tNr[0], tNc[1], tNr[0],tNc[0]);

  
#else



    // TODO: variables for better readability instead of tNr[i], tNc[i]
     START_TIMING_MEASUREMENT("Separable Inverse cdf_97");

    for (int i = levels-1; i >= 1; i--) {
        inverse_pass1_even_symmetric_cdf_97_wavelet(d_coeffs[0].data(), d_coeffs[3*i+1].data(), d_coeffs[3*i+2].data(), d_coeffs[3*i+3].data(), d_tmp1, d_tmp2, tNr[i+1], tNc[i+1], tNr[i], hlen);
        inverse_pass2_even_symmetric_cdf_97_wavelet(d_tmp1, d_tmp2, d_coeffs[0].data(), tNr[i], tNc[i+1], tNc[i], hlen);


    }
    // First scale
    inverse_pass1_even_symmetric_cdf_97_wavelet(d_coeffs[0].data(), d_coeffs[1].data(), d_coeffs[2].data(), d_coeffs[3].data(), d_tmp1, d_tmp2, tNr[1], tNc[1], tNr[0], hlen);
    inverse_pass2_even_symmetric_cdf_97_wavelet(d_tmp1, d_tmp2, d_image, tNr[0], tNc[1], tNc[0], hlen);

    STOP_TIMING_MEASUREMENT;
#endif

    return 0;
}

int separable_wavelet_transform::inverse_2D_even_symmetric_cdf_53_wavelet(DTYPE* d_image, wavelet_coeff_t& d_coeffs)
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
    DTYPE* d_tmp1 = tmp.data();
    DTYPE* d_tmp2 = tmp.data() + Nr*tNc[1];


#ifdef TIMING
  
     DTYPE* d_tmp1_pass1 = d_tmp1;
     DTYPE* d_tmp2_pass1 = d_tmp2;
  
     for (int i = levels-1; i >= 1; i--) {
        double avg_pass1{0};
        double avg_pass2{0};
        double var_pass1{0};
        double var_pass2{0};
        double K1,K2;
        aligned_data_vec_t temp_coeffs0( d_coeffs[0].begin(), d_coeffs[0].end());    
        for(int j=0;j<100;++j)
        {   
  	      d_tmp1_pass1 = d_tmp1;
          d_tmp2_pass1 = d_tmp2;
          std::copy( d_coeffs[0].begin(), d_coeffs[0].end(),temp_coeffs0.begin());
          auto t1 = measure_type::duration(&separable_wavelet_transform::inverse_pass1_even_symmetric_cdf_53_wavelet,this,temp_coeffs0.data(), d_coeffs[3*i+1].data(), d_coeffs[3*i+2].data(), d_coeffs[3*i+3].data(), d_tmp1_pass1, d_tmp2_pass1, tNr[i+1], tNc[i+1], tNr[i], hlen);
//          auto t2 = measure_type::duration(&separable_wavelet_transform::inverse_pass2_even_symmetric_cdf_53_wavelet,this,d_tmp1_pass1, d_tmp2_pass1, d_coeffs[0].data(), tNr[i], tNc[i+1], tNc[i], hlen);
          auto t2 = measure_type::duration(&separable_wavelet_transform::inverse_pass2_even_symmetric_cdf_53_wavelet,this,d_tmp1_pass1, d_tmp2_pass1, temp_coeffs0.data(), tNr[i], tNc[i+1], tNc[i], hlen);
           if(j==0)
           {
            K1 = t1.count();
            K2 = t2.count();
           }
           else
           {
		     avg_pass1 += (t1.count()-K1); 
		     avg_pass2 += (t2.count()-K2); 
		     var_pass1 += (t1.count()-K1)*(t1.count()-K1); 
		     var_pass2 += (t2.count()-K2)*(t2.count()-K2); 
	       }

        }  
        d_tmp1 = d_tmp1_pass1;
        d_tmp2 = d_tmp2_pass1;
        std::copy(temp_coeffs0.begin(),temp_coeffs0.end(),d_coeffs[0].begin());
   	    double var1 = (var_pass1-(avg_pass1*avg_pass1)/double(100))/double(100-1);
	    double var2 = (var_pass2-(avg_pass2*avg_pass2)/double(100))/double(100-1);

	    double avg1 = K1 + avg_pass1/double(100);
	    double avg2 = K2 + avg_pass2/double(100);

       // d_coeffs0_pass1 = d_coeffs[0].data();
        create_output("inverse_pass1_even_symmetric_cdf_53_wavelet",i,avg1,var1,tNr[i+1], tNc[i+1],tNr[i],tNc[i+1]);
        create_output("inverse_pass2_even_symmetric_cdf_53_wavelet",i,avg2,var2,tNr[i],tNc[i+1],tNr[i],tNc[i]);
 
   	    //std::cout << "separable_wavelet_transform::inverse_pass1_even_symmetric_cdf_53_wavelet level " << i << " : "<< avg_pass1.count()/100. << measure_result_unit << std::endl;
	    //std::cout << "separable_wavelet_transform::inverse_pass1_even_symmetric_cdf_53_wavelet level " << i << " : "<< avg_pass2.count()/100. << measure_result_unit << std::endl;
    }
    
    // First scale
    double avg_pass1{0};
    double avg_pass2{0};
    double var_pass1{0};
    double var_pass2{0};
    double K1,K2;

    for(int i=0;i<100;++i)
    {
      auto t1 = measure_type::duration(&separable_wavelet_transform::inverse_pass1_even_symmetric_cdf_53_wavelet,this,d_coeffs[0].data(), d_coeffs[1].data(), d_coeffs[2].data(), d_coeffs[3].data(), d_tmp1, d_tmp2, tNr[1], tNc[1], tNr[0], hlen);
      auto t2 = measure_type::duration(&separable_wavelet_transform::inverse_pass2_even_symmetric_cdf_53_wavelet,this,d_tmp1, d_tmp2, d_image, tNr[0], tNc[1], tNc[0], hlen);
     if(i==0)
     {
       K1 = t1.count();
       K2 = t2.count();
     }
     else
     {
		avg_pass1 += (t1.count()-K1); 
		avg_pass2 += (t2.count()-K2); 
		var_pass1 += (t1.count()-K1)*(t1.count()-K1); 
		var_pass2 += (t2.count()-K2)*(t2.count()-K2); 
	 }

    } 

	 double var1 = (var_pass1-(avg_pass1*avg_pass1)/double(100))/double(100-1);
	 double var2 = (var_pass2-(avg_pass2*avg_pass2)/double(100))/double(100-1);

	 double avg1 = K1 + avg_pass1/double(100);
	 double avg2 = K2 + avg_pass2/double(100);
  
  
    create_output("inverse_pass1_even_symmetric_cdf_53_wavelet",0,avg1,var1,tNr[1], tNc[1], tNr[0],tNc[1]);
    create_output("inverse_pass2_even_symmetric_cdf_53_wavelet",0,avg2,var2,tNr[0], tNc[1], tNr[0],tNc[0]);
 
	// std::cout << "separable_wavelet_transform::inverse_pass1_even_symmetric_cdf_53_wavelet level 0 : " << avg_pass1.count()/100. << measure_result_unit << std::endl;
	// std::cout << "separable_wavelet_transform::inverse_pass2_even_symmetric_cdf_53_wavelet level 0 : " << avg_pass2.count()/100. << measure_result_unit << std::endl;

 

#else



    // TODO: variables for better readability instead of tNr[i], tNc[i]
     START_TIMING_MEASUREMENT("Separable Inverse cdf_53");

    for (int i = levels-1; i >= 1; i--) {
        inverse_pass1_even_symmetric_cdf_53_wavelet(d_coeffs[0].data(), d_coeffs[3*i+1].data(), d_coeffs[3*i+2].data(), d_coeffs[3*i+3].data(), d_tmp1, d_tmp2, tNr[i+1], tNc[i+1], tNr[i], hlen);
        inverse_pass2_even_symmetric_cdf_53_wavelet(d_tmp1, d_tmp2, d_coeffs[0].data(), tNr[i], tNc[i+1], tNc[i], hlen);


    }
    // First scale
    inverse_pass1_even_symmetric_cdf_53_wavelet(d_coeffs[0].data(), d_coeffs[1].data(), d_coeffs[2].data(), d_coeffs[3].data(), d_tmp1, d_tmp2, tNr[1], tNc[1], tNr[0], hlen);
    inverse_pass2_even_symmetric_cdf_53_wavelet(d_tmp1, d_tmp2, d_image, tNr[0], tNc[1], tNc[0], hlen);

    STOP_TIMING_MEASUREMENT;
#endif

    return 0;
}

    
int separable_wavelet_transform::inverse_1D(DTYPE* d_image,wavelet_coeff_t& d_coeffs)
{
	int Nr = winfos.Nr, Nc = winfos.Nc, levels = winfos.nlevels, hlen = winfos.hlen;
    // Table of sizes. FIXME: consider adding this in the w_info structure
    int tNc[levels+1]; tNc[0] = Nc;
    for (int i = 1; i <= levels; i++) {
        tNc[i] = tNc[i-1];
        w_div2(tNc + i);
    }
    DTYPE* d_tmp1, *d_tmp2;
    d_tmp1 = d_coeffs[0].data();
    d_tmp2 = tmp.data();

    for (int i = levels-1; i >= 1; i--) {
        inverse_pass2(d_tmp1, d_coeffs[i+1].data(), d_tmp2, Nr, tNc[i+1], tNc[i], hlen);
        w_swap_ptr(&d_tmp1, &d_tmp2);
    }
    if ((levels > 1) && ((levels & 1) == 0))
    {
		 std::memcpy(d_coeffs[0].data(), d_tmp1, Nr*tNc[1]*sizeof(DTYPE));
		 
    }
    // First scale
   inverse_pass2(d_coeffs[0].data(), d_coeffs[1].data(), d_image, Nr, tNc[1], Nc, hlen);

    return 0;
}

int separable_wavelet_transform::forward_swt_2D(DTYPE* d_image, wavelet_coeff_t& d_coeffs)
{
    int Nr = winfos.Nr, Nc = winfos.Nc, levels = winfos.nlevels, hlen = winfos.hlen;

    DTYPE* d_tmp1 = tmp.data();
    DTYPE* d_tmp2 = tmp.data() + Nr*Nc;

    // First level
    forward_swt_pass1(d_image, d_tmp1, d_tmp2, Nr, Nc, hlen, 1);
    forward_swt_pass2(d_tmp1, d_tmp2, d_coeffs[0].data(), d_coeffs[1].data(), d_coeffs[2].data(), d_coeffs[3].data(), Nr, Nc, hlen, 1);
    // Other levels
    for (int i=1; i < levels; i++) {
        forward_swt_pass1(d_coeffs[0].data(), d_tmp1, d_tmp2, Nr, Nc, hlen, i+1);
        forward_swt_pass2(d_tmp1, d_tmp2, d_coeffs[0].data(), d_coeffs[3*i+1].data(), d_coeffs[3*i+2].data(), d_coeffs[3*i+3].data(), Nr, Nc, hlen, i+1);
    }
    return 0;
	
}

int separable_wavelet_transform::forward_swt_1D(DTYPE* d_image,wavelet_coeff_t& d_coeffs)
{
	int Nr = winfos.Nr, Nc = winfos.Nc, levels = winfos.nlevels, hlen = winfos.hlen;

    DTYPE* d_tmp1 = d_coeffs[0].data();
    DTYPE* d_tmp2 = tmp.data();

    // First level
    forward_swt_pass1(d_image, d_coeffs[0].data(), d_coeffs[1].data(), Nr, Nc, hlen, 1);
    // Other levels
    for (int i=1; i < levels; i++) {
        forward_swt_pass1(d_tmp1, d_tmp2, d_coeffs[i+1].data(), Nr, Nc, hlen, i+1);
        w_swap_ptr(&d_tmp1, &d_tmp2);
    }
    if ((levels & 1) == 0)
    {
		std::memcpy(d_coeffs[0].data(),tmp.data(), Nr*Nc*sizeof(DTYPE));
		
    }
    return 0;
}

int separable_wavelet_transform::inverse_swt_2D(DTYPE* d_image, wavelet_coeff_t& d_coeffs)
{
	int Nr = winfos.Nr, Nc = winfos.Nc, levels = winfos.nlevels, hlen = winfos.hlen;
    DTYPE* d_tmp1 = tmp.data();
    DTYPE* d_tmp2 = tmp.data() + Nr*Nc;

    for (int i = levels-1; i >= 1; i--) {
        inverse_swt_pass1(d_coeffs[0].data(), d_coeffs[3*i+1].data(), d_coeffs[3*i+2].data(), d_coeffs[3*i+3].data(), d_tmp1, d_tmp2, Nr, Nc, hlen, i+1);
        inverse_swt_pass2(d_tmp1, d_tmp2, d_coeffs[0].data(), Nr, Nc, hlen, i+1);
    }
    // First scale
    inverse_swt_pass1(d_coeffs[0].data(), d_coeffs[1].data(), d_coeffs[2].data(), d_coeffs[3].data(), d_tmp1, d_tmp2, Nr, Nc, hlen, 1);
    inverse_swt_pass2(d_tmp1, d_tmp2, d_image, Nr, Nc, hlen, 1);

    return 0;
}
int separable_wavelet_transform::inverse_swt_1D(DTYPE* d_image,wavelet_coeff_t& d_coeffs)
{
	int Nr = winfos.Nr, Nc = winfos.Nc, levels = winfos.nlevels, hlen = winfos.hlen;
    DTYPE* d_tmp1 = d_coeffs[0].data();
    DTYPE* d_tmp2 = tmp.data();


    for (int i = levels-1; i >= 1; i--) {
        inverse_swt_pass2(d_tmp1, d_coeffs[i+1].data(), d_tmp2, Nr, Nc, hlen, i+1);
        w_swap_ptr(&d_tmp1, &d_tmp2);
    }
    if ((levels & 1) == 0)
    {
		std::memcpy(d_coeffs[0].data(),tmp.data(), Nr*Nc*sizeof(DTYPE));
		
    }
    // First scale
    inverse_swt_pass2(d_coeffs[0].data(), d_coeffs[1].data(), d_image, Nr, Nc, hlen, 1);

    return 0;
}

void separable_wavelet_transform::forward_pass1(DTYPE* __restrict__ img, DTYPE* __restrict__ tmp_a1, DTYPE* __restrict__ tmp_a2, int Nr, int Nc, int hlen)
{
 int Nc_is_odd = (Nc & 1);
 int Nc2 = (Nc + Nc_is_odd)/2;
 int c, hL, hR, filter_len;
 if (hlen & 1) { // odd kernel size
     c = hlen/2;
     hL = c;
     hR = c;
     //std::cout << "odd kernel size" << std::endl;
  }
  else { // even kernel size : center is shifted to the left
    c = hlen/2 - 1;
    hL = c;
    hR = c+1;
     //std::cout << "even kernel size" << std::endl;
  }
  
  
  filter_len = hL+hR; 
  DTYPE*  c_L =  c_kern_L.data();
  DTYPE*  c_H =  c_kern_H.data();
  
 #pragma omp parallel for schedule(static)
 for(int row=0;row<Nr;++row)
  {
	  //left
	  for(int col=0;col<filter_len;++col)
	  {
        DTYPE res_tmp_a1 = 0, res_tmp_a2 = 0;
        DTYPE img_val;

        // Convolution with periodic boundaries extension.
        for (int jx = 0; jx <= hR+hL; jx++) {

            int idx_x = col*2 - c + jx;

            if (idx_x < 0) idx_x += (Nc + Nc_is_odd); // if N is odd, image is virtually extended
            // no "else if", since idx_x can be > N-1  after being incremented
            if (idx_x > Nc-1) {
                if ((idx_x == Nc) && (Nc_is_odd)) idx_x--; // if N is odd, repeat the right-most element
                else idx_x -= (Nc + Nc_is_odd); // if N is odd, image is virtually extended
            }

            img_val = img[row*Nc + idx_x];
            res_tmp_a1 += img_val * c_kern_L[hlen-1 - jx];
            res_tmp_a2 += img_val * c_kern_H[hlen-1 - jx];

        }
        tmp_a1[row* Nc2 + col] = res_tmp_a1;
        tmp_a2[row* Nc2 + col] = res_tmp_a2; 		  
	  }

	  for(int col=filter_len;col<Nc2-filter_len;++col)
	  {
        DTYPE res_tmp_a1 = 0, res_tmp_a2 = 0;
        DTYPE img_val;


        // Convolution with periodic boundaries extension.
        #pragma omp simd aligned(img,c_L,c_H:16)
        for (int jx = 0; jx <= hR+hL; jx++) {

            int idx_x = col*2 - c + jx;

            img_val = img[row*Nc + idx_x];
            res_tmp_a1 += img_val * c_L[hlen-1 - jx];
            res_tmp_a2 += img_val * c_H[hlen-1 - jx];

        }
        tmp_a1[row* Nc2 + col] = res_tmp_a1;
        tmp_a2[row* Nc2 + col] = res_tmp_a2; 		  
	  }

	  for(int col=Nc2-filter_len;col<Nc2;++col)
	  {
        DTYPE res_tmp_a1 = 0, res_tmp_a2 = 0;
        DTYPE img_val;

        // Convolution with periodic boundaries extension.
        for (int jx = 0; jx <= hR+hL; jx++) {

            int idx_x = col*2 - c + jx;

            if (idx_x < 0) idx_x += (Nc + Nc_is_odd); // if N is odd, image is virtually extended
            // no "else if", since idx_x can be > N-1  after being incremented
            if (idx_x > Nc-1) {
                if ((idx_x == Nc) && (Nc_is_odd)) idx_x--; // if N is odd, repeat the right-most element
                else idx_x -= (Nc + Nc_is_odd); // if N is odd, image is virtually extended
            }

            img_val = img[row*Nc + idx_x];
            res_tmp_a1 += img_val * c_kern_L[hlen-1 - jx];
            res_tmp_a2 += img_val * c_kern_H[hlen-1 - jx];

        }
        tmp_a1[row* Nc2 + col] = res_tmp_a1;
        tmp_a2[row* Nc2 + col] = res_tmp_a2; 		  
	  }


  }


/*
 int Nc_is_odd = (Nc & 1);
 int Nc2 = (Nc + Nc_is_odd)/2;
   
std::cout << "Forward separable path 1 rows: " << Nr << " cols: " << Nc2<< std::endl; 
   
// #pragma omp parallel for
 for(int row=0;row<Nr;++row)
  {
	  for(int col=0;col<Nc2;++col)
	  {
         int c, hL, hR;
         if (hlen & 1) { // odd kernel size
            c = hlen/2;
            hL = c;
            hR = c;
           // std::cout << "ODD" << std::endl;
        }
        else { // even kernel size : center is shifted to the left
            c = hlen/2 - 1;
            hL = c;
            hR = c+1;
        }
        DTYPE res_tmp_a1 = 0, res_tmp_a2 = 0;
        DTYPE img_val;

        // Convolution with periodic boundaries extension.
        for (int jx = 0; jx <= hR+hL; jx++) {

            int idx_x = col*2 - c + jx;

            if (idx_x < 0) idx_x += (Nc + Nc_is_odd); // if N is odd, image is virtually extended
            // no "else if", since idx_x can be > N-1  after being incremented
            if (idx_x > Nc-1) {
                if ((idx_x == Nc) && (Nc_is_odd)) idx_x--; // if N is odd, repeat the right-most element
                else idx_x -= (Nc + Nc_is_odd); // if N is odd, image is virtually extended
            }

            img_val = img[row*Nc + idx_x];
            res_tmp_a1 += img_val * c_kern_L[hlen-1 - jx];
            res_tmp_a2 += img_val * c_kern_H[hlen-1 - jx];

        }
        tmp_a1[row* Nc2 + col] = res_tmp_a1;
        tmp_a2[row* Nc2 + col] = res_tmp_a2; 		  
	  }
  }
   */	
}


void separable_wavelet_transform::forward_pass1_even_symmetric(DTYPE* __restrict__ img, DTYPE* __restrict__ tmp_a1, DTYPE* __restrict__ tmp_a2, int Nr, int Nc, int hlen)
{
  int Nc_is_odd = (Nc & 1);
  int Nc2 = (Nc + Nc_is_odd)/2;
  int c, hL, hR, filter_len;
  c = hlen/2 - 1;
  hL = c;
  hR = c+1;
  
  
  filter_len = hL+hR; 
  DTYPE*  c_L =  c_kern_L.data();
  DTYPE*  c_H =  c_kern_H.data();

  DTYPE* img1;
  DTYPE* img2;
  DTYPE res_tmp_a1, res_tmp_a2;
  DTYPE img_val;
  
  #pragma omp parallel for private(img1,img2,res_tmp_a1,res_tmp_a2,img_val) schedule(static)
  for(int row=0;row<Nr;++row)
  {
	
 
	  for(int col=0;col<filter_len;++col)
	  {

        res_tmp_a1 = 0;
        res_tmp_a2 = 0;

        // Convolution with periodic boundaries extension.
        for (int jx = 0; jx <= hR+hL; jx++) {

            int idx_x = col*2 - c + jx;

            if (idx_x < 0) idx_x += (Nc + Nc_is_odd); // if N is odd, image is virtually extended
            // no "else if", since idx_x can be > N-1  after being incremented
            if (idx_x > Nc-1) {
                if ((idx_x == Nc) && (Nc_is_odd)) idx_x--; // if N is odd, repeat the right-most element
                else idx_x -= (Nc + Nc_is_odd); // if N is odd, image is virtually extended
            }

            img_val = img[row*Nc + idx_x];
            res_tmp_a1 += img_val * c_kern_L[hlen-1 - jx];
            res_tmp_a2 += img_val * c_kern_H[hlen-1 - jx];

        }
        tmp_a1[row* Nc2 + col] = res_tmp_a1;
        tmp_a2[row* Nc2 + col] = res_tmp_a2; 		  
	  }

	  for(int col=filter_len;col<Nc2-filter_len;++col)
	  {
 
       
         img1 = &img[row*Nc + col*2 - c];
         img2 = &img[row*Nc + col*2 + c];
         res_tmp_a1 = *(img2+1)*c_L[0]+ *(img1+c)*c_L[hR];
         res_tmp_a2 = *(img1)*c_H[hR+hL]+*(img1+c+1)*c_H[hR-1];
 
        // Convolution with periodic boundaries extension.
        #pragma GCC ivdep
        for (int jx = 0; jx < hL; jx++) {
            res_tmp_a1 += (*(img1)+ *(img2))* c_L[hlen-1 - jx];
            res_tmp_a2 += (*(img1+1)+ *(img2+1)) * c_H[hlen-1 - jx-1];
        /* if(col==filter_len && row == 10)
         {
			 std::cout << "even loop " << idx_xq <<" index2 " << (idx_x1q) << " im0 " << img[row*Nc + idx_xq] << " im1 " << img[row*Nc + idx_x1q] << " c_l " 
			 << c_L[hlen-1 - jx] << " c_h " <<  c_H[hlen-1 - jx]<< " c_index " <<  (hlen-1 - jx)
			 << " r0 " << res_tmp_a1q <<  " r1 " << res_tmp_a2q << std::endl;
		 }*/
            ++img1;
            --img2;
        }


       
        
/*
         DTYPE res_tmp_a1 = 0;
         DTYPE res_tmp_a2 = 0;
         DTYPE img_val;

        for (int jx = 0; jx <= hR+hL; jx++) {

            int idx_x = col*2 - c + jx;

            img_val = img[row*Nc + idx_x];
            res_tmp_a1 += img_val * c_L[hlen-1 - jx];
            res_tmp_a2 += img_val * c_H[hlen-1 - jx];
         if(col==filter_len && row == 10)
         {
			 std::cout << "even loop full " << idx_x << " im " << img[row*Nc + idx_x] << " c_l "
			 << c_L[hlen-1 - jx] << " c_h " <<  c_H[hlen-1 - jx] << " c_index " <<  (hlen-1 - jx) 
			 << " r0 " << res_tmp_a1 <<  " r1 " << res_tmp_a2 << std::endl;
		 }

        }*/
 
        tmp_a1[row* Nc2 + col] = res_tmp_a1;
        tmp_a2[row* Nc2 + col] = res_tmp_a2; 		  
	  }

	  for(int col=Nc2-filter_len;col<Nc2;++col)
	  {
        res_tmp_a1 = 0;
        res_tmp_a2 = 0;

        // Convolution with periodic boundaries extension.
        for (int jx = 0; jx <= hR+hL; jx++) {

            int idx_x = col*2 - c + jx;

            if (idx_x < 0) idx_x += (Nc + Nc_is_odd); // if N is odd, image is virtually extended
            // no "else if", since idx_x can be > N-1  after being incremented
            if (idx_x > Nc-1) {
                if ((idx_x == Nc) && (Nc_is_odd)) idx_x--; // if N is odd, repeat the right-most element
                else idx_x -= (Nc + Nc_is_odd); // if N is odd, image is virtually extended
            }

            img_val = img[row*Nc + idx_x];
            res_tmp_a1 += img_val * c_kern_L[hlen-1 - jx];
            res_tmp_a2 += img_val * c_kern_H[hlen-1 - jx];

        }
        tmp_a1[row* Nc2 + col] = res_tmp_a1;
        tmp_a2[row* Nc2 + col] = res_tmp_a2; 		  
	  }


  }



}


void separable_wavelet_transform::forward_pass1_even_symmetric_cdf_53_wavelet(const DTYPE* __restrict__ img, DTYPE* __restrict__ tmp_a1, DTYPE* __restrict__ tmp_a2, int Nr, int Nc, int hlen)
{
  const int Nc_is_odd = (Nc & 1);
  const int Nc2 = (Nc + Nc_is_odd)/2;
  //int c, hL, hR, filter_len;
  const int c= 2;//hlen/2 - 1;
  const int hL = 2;//c;
  const int hR = 3;//c+1;
  const int filter_len = 5; 
  
  
  const aligned_data_vec_t filter1={0.0,-1.0/8.,2.0/8.,6.0/8.,2.0/8.,-1.0/8.}; 
  
  const aligned_data_vec_t filter2={0.,-0.5,1.0,-0.5,0.,0.};
  
  
  
  const DTYPE*  __restrict__ c_L =  filter1.data();
  const DTYPE*  __restrict__ c_H =  filter2.data();

  const DTYPE* __restrict__ img1;
  const DTYPE* __restrict__ img2;
  
  DTYPE res_tmp_a1, res_tmp_a2;
  DTYPE img_val;
  


  
  
   #pragma omp parallel for private(img1,img2,res_tmp_a1,res_tmp_a2,img_val) schedule(static,128)
  for(int row=0;row<Nr;++row)
  {
	
 
	  for(int col=0;col<filter_len;++col)
	  {

        res_tmp_a1 = 0;
        res_tmp_a2 = 0;

        // Convolution with periodic boundaries extension.
        for (int jx = 0; jx <= hR+hL; jx++) {

            int idx_x = col*2 - c + jx;

            if (idx_x < 0) idx_x += (Nc + Nc_is_odd); // if N is odd, image is virtually extended
            // no "else if", since idx_x can be > N-1  after being incremented
            if (idx_x > Nc-1) {
                if ((idx_x == Nc) && (Nc_is_odd)) idx_x--; // if N is odd, repeat the right-most element
                else idx_x -= (Nc + Nc_is_odd); // if N is odd, image is virtually extended
            }

            img_val = img[row*Nc + idx_x];
            res_tmp_a1 += img_val * c_kern_L[hlen-1 - jx];
            res_tmp_a2 += img_val * c_kern_H[hlen-1 - jx];

        }
        tmp_a1[row* Nc2 + col] = res_tmp_a1;
        tmp_a2[row* Nc2 + col] = res_tmp_a2; 		  
	  }

	  for(int col=filter_len;col<Nc2-filter_len;++col)
	  {
 
       
         img1 = &img[row*Nc + col*2 - c];
         img2 = &img[row*Nc + col*2 + c];
         res_tmp_a1 = *(img2+1)*c_L[0]+ *(img1+c)*c_L[hR];
         res_tmp_a2 = *(img1)*c_H[hR+hL]+*(img1+c+1)*c_H[hR-1];
 
      
        #pragma GCC ivdep
        for (int jx = 0; jx < hL; jx++) {
            res_tmp_a1 += (*(img1)+ *(img2))* c_L[hlen-1 - jx];
            res_tmp_a2 += (*(img1+1)+ *(img2+1)) * c_H[hlen-1 - jx-1];
            ++img1;
            --img2;
        }

/*            res_tmp_a1 += (*(img1)+ *(img2))* c_L[hlen-1];
            res_tmp_a2 += (*(img1+1)+ *(img2+1)) * c_H[hlen-2];
            ++img1;
            --img2;
            res_tmp_a1 += (*(img1)+ *(img2))* c_L[hlen-2];
            res_tmp_a2 += (*(img1+1)+ *(img2+1)) * c_H[hlen-3];
            ++img1;
            --img2;
*/           
 
        tmp_a1[row* Nc2 + col] = res_tmp_a1;
        tmp_a2[row* Nc2 + col] = res_tmp_a2; 		  
	  }

	  for(int col=Nc2-filter_len;col<Nc2;++col)
	  {
        res_tmp_a1 = 0;
        res_tmp_a2 = 0;

        // Convolution with periodic boundaries extension.
        for (int jx = 0; jx <= hR+hL; jx++) {

            int idx_x = col*2 - c + jx;

            if (idx_x < 0) idx_x += (Nc + Nc_is_odd); // if N is odd, image is virtually extended
            // no "else if", since idx_x can be > N-1  after being incremented
            if (idx_x > Nc-1) {
                if ((idx_x == Nc) && (Nc_is_odd)) idx_x--; // if N is odd, repeat the right-most element
                else idx_x -= (Nc + Nc_is_odd); // if N is odd, image is virtually extended
            }

            img_val = img[row*Nc + idx_x];
            res_tmp_a1 += img_val * c_kern_L[hlen-1 - jx];
            res_tmp_a2 += img_val * c_kern_H[hlen-1 - jx];

        }
        tmp_a1[row* Nc2 + col] = res_tmp_a1;
        tmp_a2[row* Nc2 + col] = res_tmp_a2; 		  
	  }


  }


  



}


void separable_wavelet_transform::forward_pass1_even_symmetric_cdf_97_wavelet(const DTYPE* __restrict__ img, DTYPE* __restrict__ tmp_a1, DTYPE* __restrict__ tmp_a2, int Nr, int Nc, int hlen)
{
  const int Nc_is_odd = (Nc & 1);
  const int Nc2 = (Nc + Nc_is_odd)/2;
  //int c, hL, hR, filter_len;
  const int c= 4;//hlen/2 - 1;
  const int hL = 4;//c;
  const int hR = 5;//c+1;
  const int filter_len = 9; 
  
  
  const aligned_data_vec_t filter1={0.0,0.026748757411,-0.016864118443,-0.078223266529,0.266864118443,0.602949018236,0.266864118443,
                              -0.078223266529,-0.016864118443,0.026748757411 }; 
  
  const aligned_data_vec_t filter2={0.,0.091271763114,-0.057543526229,-0.591271763114,1.11508705,
                                    -0.591271763114,-0.057543526229,0.091271763114,0.,0.};
  
  
  const DTYPE*  __restrict__ c_L =  filter1.data();
  const DTYPE*  __restrict__ c_H =  filter2.data();

  const DTYPE* __restrict__ img1;
  const DTYPE* __restrict__ img2;
  
  DTYPE res_tmp_a1, res_tmp_a2;
  DTYPE img_val;
  


  
  
   #pragma omp parallel for private(img1,img2,res_tmp_a1,res_tmp_a2,img_val) schedule(static)
  for(int row=0;row<Nr;++row)
  {
	/*
      if(Nr==512 && Nc==512)
      {
		  #pragma omp critical
		  std::cout << "row " << row << " tid " << omp_get_thread_num() << "/" << omp_get_num_threads() << std::endl;
	  }*/ 
	  
	  for(int col=0;col<filter_len;++col)
	  {

        res_tmp_a1 = 0;
        res_tmp_a2 = 0;

        // Convolution with periodic boundaries extension.
        for (int jx = 0; jx <= hR+hL; jx++) {

            int idx_x = col*2 - c + jx;

            if (idx_x < 0) idx_x += (Nc + Nc_is_odd); // if N is odd, image is virtually extended
            // no "else if", since idx_x can be > N-1  after being incremented
            if (idx_x > Nc-1) {
                if ((idx_x == Nc) && (Nc_is_odd)) idx_x--; // if N is odd, repeat the right-most element
                else idx_x -= (Nc + Nc_is_odd); // if N is odd, image is virtually extended
            }

            img_val = img[row*Nc + idx_x];
            res_tmp_a1 += img_val * c_kern_L[hlen-1 - jx];
            res_tmp_a2 += img_val * c_kern_H[hlen-1 - jx];

        }
        tmp_a1[row* Nc2 + col] = res_tmp_a1;
        tmp_a2[row* Nc2 + col] = res_tmp_a2; 		  
	  }

	  for(int col=filter_len;col<Nc2-filter_len;++col)
	  {
 
       
         img1 = &img[row*Nc + col*2 - c];
         img2 = &img[row*Nc + col*2 + c+1];
         res_tmp_a1 = *(img1+c)*c_L[hR];
         res_tmp_a2 =  *(img1+c+1)*c_H[hR-1];
         const DTYPE* pL= &c_L[hlen-1];
         const DTYPE* pH= &c_H[hlen-2];
        // Convolution with periodic boundaries extension.
        #pragma GCC ivdep
        for (int jx = 0; jx < hL; jx++) {
            res_tmp_a1 += (*(img1))* *pL;
            res_tmp_a2 += (*(img2)) * *pH;
            ++img1;
            res_tmp_a2 += (*(img1)) * *pH;
            --img2;
            res_tmp_a1 += (*(img2))* *pL;
            --pL;
            --pH;
        }
    
 
        tmp_a1[row* Nc2 + col] = res_tmp_a1;
        tmp_a2[row* Nc2 + col] = res_tmp_a2; 		  
	  }

	  for(int col=Nc2-filter_len;col<Nc2;++col)
	  {
        res_tmp_a1 = 0;
        res_tmp_a2 = 0;

        // Convolution with periodic boundaries extension.
        for (int jx = 0; jx <= hR+hL; jx++) {

            int idx_x = col*2 - c + jx;

            if (idx_x < 0) idx_x += (Nc + Nc_is_odd); // if N is odd, image is virtually extended
            // no "else if", since idx_x can be > N-1  after being incremented
            if (idx_x > Nc-1) {
                if ((idx_x == Nc) && (Nc_is_odd)) idx_x--; // if N is odd, repeat the right-most element
                else idx_x -= (Nc + Nc_is_odd); // if N is odd, image is virtually extended
            }

            img_val = img[row*Nc + idx_x];
            res_tmp_a1 += img_val * c_kern_L[hlen-1 - jx];
            res_tmp_a2 += img_val * c_kern_H[hlen-1 - jx];

        }
        tmp_a1[row* Nc2 + col] = res_tmp_a1;
        tmp_a2[row* Nc2 + col] = res_tmp_a2; 		  
	  }


  }



}




void separable_wavelet_transform::forward_pass1_even_symmetric_unroll2(DTYPE* __restrict__ img, DTYPE* __restrict__ tmp_a1, DTYPE* __restrict__ tmp_a2, int Nr, int Nc, int hlen)
{
  int Nc_is_odd = (Nc & 1);
  int Nc2 = (Nc + Nc_is_odd)/2;
  int c, hL, hR, filter_len;
  c = hlen/2 - 1;
  hL = c;
  hR = c+1;
  
  
  filter_len = hL+hR; 
  DTYPE*  c_L =  c_kern_L.data();
  DTYPE*  c_H =  c_kern_H.data();

  DTYPE* img1;
  DTYPE* img2;
  DTYPE res_tmp_a1, res_tmp_a2;
  DTYPE img_val;
  
  #pragma omp parallel for private(img1,img2,res_tmp_a1,res_tmp_a2,img_val) schedule(static)
  for(int row=0;row<Nr;++row)
  {
	
 
	  for(int col=0;col<filter_len;++col)
	  {

        res_tmp_a1 = 0;
        res_tmp_a2 = 0;

        // Convolution with periodic boundaries extension.
        for (int jx = 0; jx <= hR+hL; jx++) {

            int idx_x = col*2 - c + jx;

            if (idx_x < 0) idx_x += (Nc + Nc_is_odd); // if N is odd, image is virtually extended
            // no "else if", since idx_x can be > N-1  after being incremented
            if (idx_x > Nc-1) {
                if ((idx_x == Nc) && (Nc_is_odd)) idx_x--; // if N is odd, repeat the right-most element
                else idx_x -= (Nc + Nc_is_odd); // if N is odd, image is virtually extended
            }

            img_val = img[row*Nc + idx_x];
            res_tmp_a1 += img_val * c_kern_L[hlen-1 - jx];
            res_tmp_a2 += img_val * c_kern_H[hlen-1 - jx];

        }
        tmp_a1[row* Nc2 + col] = res_tmp_a1;
        tmp_a2[row* Nc2 + col] = res_tmp_a2; 		  
	  }

	  for(int col=filter_len;col<Nc2-filter_len;++col)
	  {
 
       
         img1 = &img[row*Nc + col*2 - c];
         img2 = &img[row*Nc + col*2 + c];
         res_tmp_a1 = *(img2+1)*c_L[0]+ *(img1+c)*c_L[hR];
         res_tmp_a2 = *(img1)*c_H[hR+hL]+*(img1+c+1)*c_H[hR-1];
 
        // Convolution with periodic boundaries extension.

            res_tmp_a1 += (*(img1)+ *(img2))* c_L[hlen-1];
            res_tmp_a2 += (*(img1+1)+ *(img2+1)) * c_H[hlen-2];
            ++img1;
            --img2;
            res_tmp_a1 += (*(img1)+ *(img2))* c_L[hlen-2];
            res_tmp_a2 += (*(img1+1)+ *(img2+1)) * c_H[hlen-3];


        
/*
         DTYPE res_tmp_a1 = 0;
         DTYPE res_tmp_a2 = 0;
         DTYPE img_val;

        for (int jx = 0; jx <= hR+hL; jx++) {

            int idx_x = col*2 - c + jx;

            img_val = img[row*Nc + idx_x];
            res_tmp_a1 += img_val * c_L[hlen-1 - jx];
            res_tmp_a2 += img_val * c_H[hlen-1 - jx];
         if(col==filter_len && row == 10)
         {
			 std::cout << "even loop full " << idx_x << " im " << img[row*Nc + idx_x] << " c_l "
			 << c_L[hlen-1 - jx] << " c_h " <<  c_H[hlen-1 - jx] << " c_index " <<  (hlen-1 - jx) 
			 << " r0 " << res_tmp_a1 <<  " r1 " << res_tmp_a2 << std::endl;
		 }

        }*/
 
        tmp_a1[row* Nc2 + col] = res_tmp_a1;
        tmp_a2[row* Nc2 + col] = res_tmp_a2; 		  
	  }

	  for(int col=Nc2-filter_len;col<Nc2;++col)
	  {
        res_tmp_a1 = 0;
        res_tmp_a2 = 0;

        // Convolution with periodic boundaries extension.
        for (int jx = 0; jx <= hR+hL; jx++) {

            int idx_x = col*2 - c + jx;

            if (idx_x < 0) idx_x += (Nc + Nc_is_odd); // if N is odd, image is virtually extended
            // no "else if", since idx_x can be > N-1  after being incremented
            if (idx_x > Nc-1) {
                if ((idx_x == Nc) && (Nc_is_odd)) idx_x--; // if N is odd, repeat the right-most element
                else idx_x -= (Nc + Nc_is_odd); // if N is odd, image is virtually extended
            }

            img_val = img[row*Nc + idx_x];
            res_tmp_a1 += img_val * c_kern_L[hlen-1 - jx];
            res_tmp_a2 += img_val * c_kern_H[hlen-1 - jx];

        }
        tmp_a1[row* Nc2 + col] = res_tmp_a1;
        tmp_a2[row* Nc2 + col] = res_tmp_a2; 		  
	  }


  }



}


void separable_wavelet_transform::forward_pass1_even_symmetric_unroll4(DTYPE* __restrict__ img, DTYPE* __restrict__ tmp_a1, DTYPE* __restrict__ tmp_a2, int Nr, int Nc, int hlen)
{
  int Nc_is_odd = (Nc & 1);
  int Nc2 = (Nc + Nc_is_odd)/2;
  int c, hL, hR, filter_len;
  c = hlen/2 - 1;
  hL = c;
  hR = c+1;
  
  
  filter_len = hL+hR; 
  DTYPE*  c_L =  c_kern_L.data();
  DTYPE*  c_H =  c_kern_H.data();

  DTYPE* img1;
  DTYPE* img2;
  DTYPE res_tmp_a1, res_tmp_a2;
  DTYPE img_val;
  
  #pragma omp parallel for private(img1,img2,res_tmp_a1,res_tmp_a2,img_val) schedule(static,128)
  for(int row=0;row<Nr;++row)
  {
	
 
	  for(int col=0;col<filter_len;++col)
	  {

        res_tmp_a1 = 0;
        res_tmp_a2 = 0;

        // Convolution with periodic boundaries extension.
        for (int jx = 0; jx <= hR+hL; jx++) {

            int idx_x = col*2 - c + jx;

            if (idx_x < 0) idx_x += (Nc + Nc_is_odd); // if N is odd, image is virtually extended
            // no "else if", since idx_x can be > N-1  after being incremented
            if (idx_x > Nc-1) {
                if ((idx_x == Nc) && (Nc_is_odd)) idx_x--; // if N is odd, repeat the right-most element
                else idx_x -= (Nc + Nc_is_odd); // if N is odd, image is virtually extended
            }

            img_val = img[row*Nc + idx_x];
            res_tmp_a1 += img_val * c_kern_L[hlen-1 - jx];
            res_tmp_a2 += img_val * c_kern_H[hlen-1 - jx];

        }
        tmp_a1[row* Nc2 + col] = res_tmp_a1;
        tmp_a2[row* Nc2 + col] = res_tmp_a2; 		  
	  }

	  for(int col=filter_len;col<Nc2-filter_len;++col)
	  {
 
       
         img1 = &img[row*Nc + col*2 - c];
         img2 = &img[row*Nc + col*2 + c];
         res_tmp_a1 = *(img2+1)*c_L[0]+ *(img1+c)*c_L[hR];
         res_tmp_a2 = *(img1)*c_H[hR+hL]+*(img1+c+1)*c_H[hR-1];
 
        // Convolution with periodic boundaries extension.

/*
            res_tmp_a1 += (*(img1)+ *(img2))* c_L[hlen-1];
            res_tmp_a2 += (*(img1+1)+ *(img2+1)) * c_H[hlen-2];
            ++img1;
            --img2;
            res_tmp_a1 += (*(img1)+ *(img2))* c_L[hlen-2];
            res_tmp_a2 += (*(img1+1)+ *(img2+1)) * c_H[hlen-3];
            ++img1;
            --img2;
            res_tmp_a1 += (*(img1)+ *(img2))* c_L[hlen-3];
            res_tmp_a2 += (*(img1+1)+ *(img2+1)) * c_H[hlen-4];
            ++img1;
            --img2;
            res_tmp_a1 += (*(img1)+ *(img2))* c_L[hlen-4];
            res_tmp_a2 += (*(img1+1)+ *(img2+1)) * c_H[hlen-5];*/

            res_tmp_a1 += (*(img1))* c_L[hlen-1];
            res_tmp_a2 += (*(img1+1)) * c_H[hlen-2];
            ++img1;
            res_tmp_a1 += (*(img1))* c_L[hlen-2];
            res_tmp_a2 += (*(img1+1)) * c_H[hlen-3];
            ++img1;
            res_tmp_a1 += (*(img1))* c_L[hlen-3];
            res_tmp_a2 += (*(img1+1)) * c_H[hlen-4];
            ++img1;
            res_tmp_a1 += (*(img1))* c_L[hlen-4];
            res_tmp_a2 += (*(img1+1)) * c_H[hlen-5];

            res_tmp_a1 += (*(img2))* c_L[hlen-1];
            res_tmp_a2 += (*(img2+1)) * c_H[hlen-2];
            --img2;
            res_tmp_a1 += (*(img2))* c_L[hlen-2];
            res_tmp_a2 += (*(img2+1)) * c_H[hlen-3];
            --img2;
            res_tmp_a1 += (*(img2))* c_L[hlen-3];
            res_tmp_a2 += (*(img2+1)) * c_H[hlen-4];
            --img2;            
            res_tmp_a1 += (*(img2))* c_L[hlen-4];
            res_tmp_a2 += (*(img2+1)) * c_H[hlen-5];
 
            tmp_a1[row* Nc2 + col] = res_tmp_a1;
            tmp_a2[row* Nc2 + col] = res_tmp_a2; 		  
	  }

	  for(int col=Nc2-filter_len;col<Nc2;++col)
	  {
        res_tmp_a1 = 0;
        res_tmp_a2 = 0;

        // Convolution with periodic boundaries extension.
        for (int jx = 0; jx <= hR+hL; jx++) {

            int idx_x = col*2 - c + jx;

            if (idx_x < 0) idx_x += (Nc + Nc_is_odd); // if N is odd, image is virtually extended
            // no "else if", since idx_x can be > N-1  after being incremented
            if (idx_x > Nc-1) {
                if ((idx_x == Nc) && (Nc_is_odd)) idx_x--; // if N is odd, repeat the right-most element
                else idx_x -= (Nc + Nc_is_odd); // if N is odd, image is virtually extended
            }

            img_val = img[row*Nc + idx_x];
            res_tmp_a1 += img_val * c_kern_L[hlen-1 - jx];
            res_tmp_a2 += img_val * c_kern_H[hlen-1 - jx];

        }
        tmp_a1[row* Nc2 + col] = res_tmp_a1;
        tmp_a2[row* Nc2 + col] = res_tmp_a2; 		  
	  }


  }



}



void separable_wavelet_transform::forward_pass2(DTYPE* __restrict__ tmp_a1, DTYPE* __restrict__ tmp_a2, DTYPE* __restrict__ c_a, DTYPE* __restrict__ c_h, DTYPE* __restrict__ c_v, DTYPE* __restrict__ c_d, int Nr, int Nc, int hlen)
{
	
 //std::cout << "Nr: " << Nr<< " Nc: " << Nc << " hlen: "	<< hlen << std::endl;
 int Nr_is_odd = (Nr & 1);
 int Nr2 = (Nr + Nr_is_odd)/2;
 
 DTYPE*  c_L =  c_kern_L.data();
 DTYPE*  c_H =  c_kern_H.data();

 int c, hL, hR, filter_len;
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
  
  filter_len = hL+hR; 


  for(int row=0;row<filter_len;++row)
  {
	  for(int col=0;col<Nc;++col)
	  {
   
          DTYPE res_a = 0, res_h = 0, res_v = 0, res_d = 0;

        // Convolution with periodic boundaries extension.
          for (int jy = 0; jy <= filter_len; jy++) {
              int idx_y = row*2 - c + jy;

              if (idx_y < 0) idx_y += (Nr + Nr_is_odd); // if N is odd, image is virtually extended
              // no "else if", since idx_y can be > N-1  after being incremented
              if (idx_y > Nr-1) {
                  if ((idx_y == Nr) && (Nr_is_odd)) idx_y--; // if N is odd, repeat the right-most element
                  else idx_y -= (Nr + Nr_is_odd); // if N is odd, image is virtually extended
              }

              res_a += tmp_a1[idx_y*Nc + col] * c_kern_L[hlen-1 - jy];
              res_h += tmp_a1[idx_y*Nc + col] * c_kern_H[hlen-1 - jy];
              res_v += tmp_a2[idx_y*Nc + col] * c_kern_L[hlen-1 - jy];
              res_d += tmp_a2[idx_y*Nc + col] * c_kern_H[hlen-1 - jy];
          }

          c_a[row* Nc + col] = res_a;
          c_h[row* Nc + col] = res_h;
          c_v[row* Nc + col] = res_v;
          c_d[row* Nc + col] = res_d;
 
      }
   }

   
  #pragma omp parallel for schedule(static)
  for(int row=filter_len;row<Nr2-filter_len;++row)
  {
	  for(int col=0;col<Nc;++col)
	  {
   
          DTYPE res_a = 0, res_h = 0, res_v = 0, res_d = 0;

        // Convolution with periodic boundaries extension.
          //#pragma omp simd aligned(tmp_a1,tmp_a2,c_kern_L,c_kern_H:16)
          #pragma omp simd aligned(tmp_a1,tmp_a2,c_L,c_H:16)
          for (int jy = 0; jy <= filter_len; jy++) {
              int idx_y = row*2 - c + jy;


              res_a += tmp_a1[idx_y*Nc + col] * c_L[hlen-1 - jy];
              res_h += tmp_a1[idx_y*Nc + col] * c_H[hlen-1 - jy];
              res_v += tmp_a2[idx_y*Nc + col] * c_L[hlen-1 - jy];
              res_d += tmp_a2[idx_y*Nc + col] * c_H[hlen-1 - jy];
          }

          c_a[row* Nc + col] = res_a;
          c_h[row* Nc + col] = res_h;
          c_v[row* Nc + col] = res_v;
          c_d[row* Nc + col] = res_d;
 
      }
   }

  for(int row=Nr2-filter_len;row<Nr2;++row)
  {
	  for(int col=0;col<Nc;++col)
	  {
   
          DTYPE res_a = 0, res_h = 0, res_v = 0, res_d = 0;

        // Convolution with periodic boundaries extension.
          for (int jy = 0; jy <= filter_len; jy++) {
              int idx_y = row*2 - c + jy;

              if (idx_y < 0) idx_y += (Nr + Nr_is_odd); // if N is odd, image is virtually extended
              // no "else if", since idx_y can be > N-1  after being incremented
              if (idx_y > Nr-1) {
                  if ((idx_y == Nr) && (Nr_is_odd)) idx_y--; // if N is odd, repeat the right-most element
                  else idx_y -= (Nr + Nr_is_odd); // if N is odd, image is virtually extended
              }

              res_a += tmp_a1[idx_y*Nc + col] * c_kern_L[hlen-1 - jy];
              res_h += tmp_a1[idx_y*Nc + col] * c_kern_H[hlen-1 - jy];
              res_v += tmp_a2[idx_y*Nc + col] * c_kern_L[hlen-1 - jy];
              res_d += tmp_a2[idx_y*Nc + col] * c_kern_H[hlen-1 - jy];
          }

          c_a[row* Nc + col] = res_a;
          c_h[row* Nc + col] = res_h;
          c_v[row* Nc + col] = res_v;
          c_d[row* Nc + col] = res_d;
 
      }
   }


}


void separable_wavelet_transform::forward_pass2_even_symmetric(DTYPE* __restrict__ tmp_a1, DTYPE* __restrict__ tmp_a2, DTYPE* __restrict__ c_a, DTYPE* __restrict__ c_h, DTYPE* __restrict__ c_v, DTYPE* __restrict__ c_d, int Nr, int Nc, int hlen)
{
	
 //std::cout << "Nr: " << Nr<< " Nc: " << Nc << " hlen: "	<< hlen << std::endl;
 int Nr_is_odd = (Nr & 1);
 int Nr2 = (Nr + Nr_is_odd)/2;
 
 DTYPE*  c_L =  c_kern_L.data();
 DTYPE*  c_H =  c_kern_H.data();

 int c, hL, hR, filter_len;
  
  c = hlen/2 - 1;
  hL = c;
  hR = c+1;
  filter_len = hL+hR; 


  for(int row=0;row<filter_len;++row)
  {
	  for(int col=0;col<Nc;++col)
	  {
   
          DTYPE res_a = 0, res_h = 0, res_v = 0, res_d = 0;

        // Convolution with periodic boundaries extension.
          for (int jy = 0; jy <= filter_len; jy++) {
              int idx_y = row*2 - c + jy;

              if (idx_y < 0) idx_y += (Nr + Nr_is_odd); // if N is odd, image is virtually extended
              // no "else if", since idx_y can be > N-1  after being incremented
              if (idx_y > Nr-1) {
                  if ((idx_y == Nr) && (Nr_is_odd)) idx_y--; // if N is odd, repeat the right-most element
                  else idx_y -= (Nr + Nr_is_odd); // if N is odd, image is virtually extended
              }

              res_a += tmp_a1[idx_y*Nc + col] * c_kern_L[hlen-1 - jy];
              res_h += tmp_a1[idx_y*Nc + col] * c_kern_H[hlen-1 - jy];
              res_v += tmp_a2[idx_y*Nc + col] * c_kern_L[hlen-1 - jy];
              res_d += tmp_a2[idx_y*Nc + col] * c_kern_H[hlen-1 - jy];
          }

          c_a[row* Nc + col] = res_a;
          c_h[row* Nc + col] = res_h;
          c_v[row* Nc + col] = res_v;
          c_d[row* Nc + col] = res_d;
 
      }
   }

 
  DTYPE res_a = 0, res_h = 0, res_v = 0, res_d = 0;
  DTYPE *img1,*img2,*img3,*img4;
  
  #pragma omp parallel for private(res_a,res_h,res_v,res_d,img1,img2,img3,img4) schedule(static)
  for(int row=filter_len;row<Nr2-filter_len;++row)
  {
	  for(int col=0;col<Nc;++col)
	  {
 
        // Convolution with periodic boundaries extension.
          //#pragma omp simd aligned(tmp_a1,tmp_a2,c_kern_L,c_kern_H:16)
          //#pragma omp simd aligned(tmp_a1,tmp_a2,c_L,c_H:16)

         img1 = &tmp_a1[(2*row-c)*Nc+col];
         img2 = &tmp_a1[(2*row+c)*Nc+col];
         img3 = &tmp_a2[(2*row-c)*Nc+col];
         img4 = &tmp_a2[(2*row+c)*Nc+col];
         res_a = *(img2+Nc)*c_L[0]+ *(img1+c*Nc)*c_L[hR];
         res_h = *(img1)*c_H[0]+ *(img1+(c+1)*Nc)*c_H[hR-1];
         res_v = *(img4+Nc)*c_L[0]+ *(img3+c*Nc)*c_L[hR];
         res_d = *(img3)*c_H[0]+ *(img3+(c+1)*Nc)*c_H[hR-1];
 
        // Convolution with periodic boundaries extension.
        #pragma GCC ivdep
        for (int jy = 0; jy < hL; jy++) {
            res_a += (*(img1)+ *(img2))* c_L[hlen-1 - jy];
            res_h += (*(img1+Nc)+ *(img2+Nc))* c_H[hlen-1 - jy-1];
            res_v += (*(img3)+ *(img4))* c_L[hlen-1 - jy];
            res_d += (*(img3+Nc)+ *(img4+Nc))* c_H[hlen-1 - jy-1];

           img1 += Nc;
           img2 -= Nc;
           img3 += Nc;
           img4 -= Nc;
        }


          c_a[row* Nc + col] = res_a;
          c_h[row* Nc + col] = res_h;
          c_v[row* Nc + col] = res_v;
          c_d[row* Nc + col] = res_d;
 
      }
   }


  for(int row=Nr2-filter_len;row<Nr2;++row)
  {
	  for(int col=0;col<Nc;++col)
	  {
   
          DTYPE res_a = 0, res_h = 0, res_v = 0, res_d = 0;

        // Convolution with periodic boundaries extension.
          for (int jy = 0; jy <= filter_len; jy++) {
              int idx_y = row*2 - c + jy;

              if (idx_y < 0) idx_y += (Nr + Nr_is_odd); // if N is odd, image is virtually extended
              // no "else if", since idx_y can be > N-1  after being incremented
              if (idx_y > Nr-1) {
                  if ((idx_y == Nr) && (Nr_is_odd)) idx_y--; // if N is odd, repeat the right-most element
                  else idx_y -= (Nr + Nr_is_odd); // if N is odd, image is virtually extended
              }

              res_a += tmp_a1[idx_y*Nc + col] * c_kern_L[hlen-1 - jy];
              res_h += tmp_a1[idx_y*Nc + col] * c_kern_H[hlen-1 - jy];
              res_v += tmp_a2[idx_y*Nc + col] * c_kern_L[hlen-1 - jy];
              res_d += tmp_a2[idx_y*Nc + col] * c_kern_H[hlen-1 - jy];
          }

          c_a[row* Nc + col] = res_a;
          c_h[row* Nc + col] = res_h;
          c_v[row* Nc + col] = res_v;
          c_d[row* Nc + col] = res_d;
 
      }
   }



}


void separable_wavelet_transform::forward_pass2_even_symmetric_cdf_53_wavelet(const DTYPE* __restrict__ tmp_a1, const DTYPE* __restrict__ tmp_a2, DTYPE* __restrict__ c_a, DTYPE* __restrict__ c_h, DTYPE* __restrict__ c_v, DTYPE* __restrict__ c_d, int Nr, int Nc, int hlen)
{
	
  //std::cout << "Nr: " << Nr<< " Nc: " << Nc << " hlen: "	<< hlen << std::endl;
  const int Nr_is_odd = (Nr & 1);
  const int Nr2 = (Nr + Nr_is_odd)/2;
  //int c, hL, hR, filter_len;
  const int c= 2;//hlen/2 - 1;
  //const int hL = 2;//c;
  const int hR = 3;//c+1;
  const int filter_len = 5; 
   
  
  
  const aligned_data_vec_t filter1={0.0,-1.0/8.,2.0/8.,6.0/8.,2.0/8.,-1.0/8.}; 
  
  const aligned_data_vec_t filter2={0.,-0.5,1.0,-0.5,0.,0.};
  
  
  const DTYPE*  __restrict__ c_L =  filter1.data();
  const DTYPE*  __restrict__ c_H =  filter2.data();



  for(int row=0;row<filter_len;++row)
  {
	  for(int col=0;col<Nc;++col)
	  {
   
          DTYPE res_a = 0, res_h = 0, res_v = 0, res_d = 0;

        // Convolution with periodic boundaries extension.
          for (int jy = 0; jy <= filter_len; jy++) {
              int idx_y = row*2 - c + jy;

              if (idx_y < 0) idx_y += (Nr + Nr_is_odd); // if N is odd, image is virtually extended
              // no "else if", since idx_y can be > N-1  after being incremented
              if (idx_y > Nr-1) {
                  if ((idx_y == Nr) && (Nr_is_odd)) idx_y--; // if N is odd, repeat the right-most element
                  else idx_y -= (Nr + Nr_is_odd); // if N is odd, image is virtually extended
              }

              res_a += tmp_a1[idx_y*Nc + col] * c_kern_L[hlen-1 - jy];
              res_h += tmp_a1[idx_y*Nc + col] * c_kern_H[hlen-1 - jy];
              res_v += tmp_a2[idx_y*Nc + col] * c_kern_L[hlen-1 - jy];
              res_d += tmp_a2[idx_y*Nc + col] * c_kern_H[hlen-1 - jy];
          }

          c_a[row* Nc + col] = res_a;
          c_h[row* Nc + col] = res_h;
          c_v[row* Nc + col] = res_v;
          c_d[row* Nc + col] = res_d;
 
      }
   }

 
  DTYPE res_a = 0, res_h = 0, res_v = 0, res_d = 0;
  const DTYPE *img1,*img2,*img3,*img4;
  
  #pragma omp parallel for private(res_a,res_h,res_v,res_d,img1,img2,img3,img4) schedule(static,128)
  for(int row=filter_len;row<Nr2-filter_len;++row)
  {
	  for(int col=0;col<Nc;++col)
	  {
 
        // Convolution with periodic boundaries extension.
          //#pragma omp simd aligned(tmp_a1,tmp_a2,c_kern_L,c_kern_H:16)
          //#pragma omp simd aligned(tmp_a1,tmp_a2,c_L,c_H:16)

         img1 = &tmp_a1[(2*row-c)*Nc+col];
         img2 = &tmp_a1[(2*row+c)*Nc+col];
         img3 = &tmp_a2[(2*row-c)*Nc+col];
         img4 = &tmp_a2[(2*row+c)*Nc+col];
         res_a = *(img2+Nc)*c_L[0]+ *(img1+c*Nc)*c_L[hR];
         res_h = *(img1)*c_H[0]+ *(img1+(c+1)*Nc)*c_H[hR-1];
         res_v = *(img4+Nc)*c_L[0]+ *(img3+c*Nc)*c_L[hR];
         res_d = *(img3)*c_H[0]+ *(img3+(c+1)*Nc)*c_H[hR-1];
 
/*        // Convolution with periodic boundaries extension.
        #pragma GCC ivdep
        for (int jy = 0; jy < hL; jy++) {
            res_a += (*(img1)+ *(img2))* c_L[hlen-1 - jy];
            res_h += (*(img1+Nc)+ *(img2+Nc))* c_H[hlen-1 - jy-1];
            res_v += (*(img3)+ *(img4))* c_L[hlen-1 - jy];
            res_d += (*(img3+Nc)+ *(img4+Nc))* c_H[hlen-1 - jy-1];

           img1 += Nc;
           img2 -= Nc;
           img3 += Nc;
           img4 -= Nc;
        }*/

            res_a += (*(img1)+ *(img2))* c_L[hlen-1];
            res_h += (*(img1+Nc)+ *(img2+Nc))* c_H[hlen-2];
            res_v += (*(img3)+ *(img4))* c_L[hlen-1];
            res_d += (*(img3+Nc)+ *(img4+Nc))* c_H[hlen-2];

           img1 += Nc;
           img2 -= Nc;
           img3 += Nc;
           img4 -= Nc;

            res_a += (*(img1)+ *(img2))* c_L[hlen-2];
            res_h += (*(img1+Nc)+ *(img2+Nc))* c_H[hlen-3];
            res_v += (*(img3)+ *(img4))* c_L[hlen-2];
            res_d += (*(img3+Nc)+ *(img4+Nc))* c_H[hlen-3];

           img1 += Nc;
           img2 -= Nc;
           img3 += Nc;
           img4 -= Nc;


          c_a[row* Nc + col] = res_a;
          c_h[row* Nc + col] = res_h;
          c_v[row* Nc + col] = res_v;
          c_d[row* Nc + col] = res_d;
 
      }
   }


  for(int row=Nr2-filter_len;row<Nr2;++row)
  {
	  for(int col=0;col<Nc;++col)
	  {
   
          DTYPE res_a = 0, res_h = 0, res_v = 0, res_d = 0;

        // Convolution with periodic boundaries extension.
          for (int jy = 0; jy <= filter_len; jy++) {
              int idx_y = row*2 - c + jy;

              if (idx_y < 0) idx_y += (Nr + Nr_is_odd); // if N is odd, image is virtually extended
              // no "else if", since idx_y can be > N-1  after being incremented
              if (idx_y > Nr-1) {
                  if ((idx_y == Nr) && (Nr_is_odd)) idx_y--; // if N is odd, repeat the right-most element
                  else idx_y -= (Nr + Nr_is_odd); // if N is odd, image is virtually extended
              }

              res_a += tmp_a1[idx_y*Nc + col] * c_kern_L[hlen-1 - jy];
              res_h += tmp_a1[idx_y*Nc + col] * c_kern_H[hlen-1 - jy];
              res_v += tmp_a2[idx_y*Nc + col] * c_kern_L[hlen-1 - jy];
              res_d += tmp_a2[idx_y*Nc + col] * c_kern_H[hlen-1 - jy];
          }

          c_a[row* Nc + col] = res_a;
          c_h[row* Nc + col] = res_h;
          c_v[row* Nc + col] = res_v;
          c_d[row* Nc + col] = res_d;
 
      }
   }





}


void separable_wavelet_transform::forward_pass2_even_symmetric_cdf_97_wavelet(const DTYPE* __restrict__ tmp_a1, const DTYPE* __restrict__ tmp_a2, DTYPE* __restrict__ c_a, DTYPE* __restrict__ c_h, DTYPE* __restrict__ c_v, DTYPE* __restrict__ c_d, int Nr, int Nc, int hlen)
{
	
 //std::cout << "Nr: " << Nr<< " Nc: " << Nc << " hlen: "	<< hlen << std::endl;
  const int Nr_is_odd = (Nr & 1);
  const int Nr2 = (Nr + Nr_is_odd)/2;
  //int c, hL, hR, filter_len;
  const int c= 4;//hlen/2 - 1;
  //const int hL = 4;//c;
  const int hR = 5;//c+1;
  const int filter_len = 9; 
   
  const aligned_data_vec_t filter1={0.0,0.026748757411,-0.016864118443,-0.078223266529,0.266864118443,0.602949018236,0.266864118443,
                              -0.078223266529,-0.016864118443,0.026748757411 }; 
  
  const aligned_data_vec_t filter2={0.,0.091271763114,-0.057543526229,-0.591271763114,1.11508705,
                                    -0.591271763114,-0.057543526229,0.091271763114,0.,0.};
  
  
  const DTYPE*  __restrict__ c_L =  filter1.data();
  const DTYPE*  __restrict__ c_H =  filter2.data();

  

  for(int row=0;row<filter_len;++row)
  {
	  for(int col=0;col<Nc;++col)
	  {
   
          DTYPE res_a = 0, res_h = 0, res_v = 0, res_d = 0;

        // Convolution with periodic boundaries extension.
          for (int jy = 0; jy <= filter_len; jy++) {
              int idx_y = row*2 - c + jy;

              if (idx_y < 0) idx_y += (Nr + Nr_is_odd); // if N is odd, image is virtually extended
              // no "else if", since idx_y can be > N-1  after being incremented
              if (idx_y > Nr-1) {
                  if ((idx_y == Nr) && (Nr_is_odd)) idx_y--; // if N is odd, repeat the right-most element
                  else idx_y -= (Nr + Nr_is_odd); // if N is odd, image is virtually extended
              }

              res_a += tmp_a1[idx_y*Nc + col] * c_kern_L[hlen-1 - jy];
              res_h += tmp_a1[idx_y*Nc + col] * c_kern_H[hlen-1 - jy];
              res_v += tmp_a2[idx_y*Nc + col] * c_kern_L[hlen-1 - jy];
              res_d += tmp_a2[idx_y*Nc + col] * c_kern_H[hlen-1 - jy];
          }

          c_a[row* Nc + col] = res_a;
          c_h[row* Nc + col] = res_h;
          c_v[row* Nc + col] = res_v;
          c_d[row* Nc + col] = res_d;
 
      }
   }

 
  DTYPE res_a = 0, res_h = 0, res_v = 0, res_d = 0;
  const DTYPE *img1,*img2,*img3,*img4;
  
  #pragma omp parallel for private(res_a,res_h,res_v,res_d,img1,img2,img3,img4) schedule(static)
  for(int row=filter_len;row<Nr2-filter_len;++row)
  {
	  img1 = &tmp_a1[(2*row-c)*Nc];
      img2 = &tmp_a1[(2*row+c)*Nc];
      img3 = &tmp_a2[(2*row-c)*Nc];
      img4 = &tmp_a2[(2*row+c)*Nc];
//      const DTYPE *pL,*pH;
	  for(int col=0;col<Nc;++col)
	  {

 
        // Convolution with periodic boundaries extension.
          //#pragma omp simd aligned(tmp_a1,tmp_a2,c_kern_L,c_kern_H:16)
          //#pragma omp simd aligned(tmp_a1,tmp_a2,c_L,c_H:16)

       
         res_a = *(img1+c*Nc)*c_L[hR];
         res_h = *(img1+(c+1)*Nc)*c_H[hR-1];
         res_v = *(img3+c*Nc)*c_L[hR];
         res_d = *(img3+(c+1)*Nc)*c_H[hR-1];
 
        // Convolution with periodic boundaries extension.
        img2 += Nc;
        img4 += Nc;
/*        #pragma GCC ivdep
        for (int jy = 0; jy < hL; jy++) {
            res_a += *(img1) * c_L[hlen-1 - jy];
            res_h += *(img2) * c_H[hlen-1 - jy-1];
            res_v += *(img3) * c_L[hlen-1 - jy];
            res_d += *(img4) * c_H[hlen-1 - jy-1];

            img1 += Nc;
            res_h += *(img1)* c_H[hlen-1 - jy-1];
            img2 -= Nc;
            res_a += (*(img2))* c_L[hlen-1 - jy];
            img3 += Nc;
            res_d += (*(img3))* c_H[hlen-1 - jy-1];
            img4 -= Nc;
            res_v += (*(img4))* c_L[hlen-1 - jy];
        }*/
     

            res_a += *(img1) * c_L[hlen-1];
            res_h += *(img2) * c_H[hlen-2];
            res_v += *(img3) * c_L[hlen-1];
            res_d += *(img4) * c_H[hlen-2];

            img1 += Nc;
            res_h += *(img1)* c_H[hlen-2];
            img2 -= Nc;
            res_a += (*(img2))* c_L[hlen-1];
            img3 += Nc;
            res_d += (*(img3))* c_H[hlen-2];
            img4 -= Nc;
            res_v += (*(img4))* c_L[hlen-1];

            res_a += *(img1) * c_L[hlen-2];
            res_h += *(img2) * c_H[hlen-3];
            res_v += *(img3) * c_L[hlen-2];
            res_d += *(img4) * c_H[hlen-3];

            img1 += Nc;
            res_h += *(img1)* c_H[hlen-3];
            img2 -= Nc;
            res_a += (*(img2))* c_L[hlen-2];
            img3 += Nc;
            res_d += (*(img3))* c_H[hlen-3];
            img4 -= Nc;
            res_v += (*(img4))* c_L[hlen-2];

            res_a += *(img1) * c_L[hlen-3];
            res_h += *(img2) * c_H[hlen-4];
            res_v += *(img3) * c_L[hlen-3];
            res_d += *(img4) * c_H[hlen-4];

            img1 += Nc;
            res_h += *(img1)* c_H[hlen-4];
            img2 -= Nc;
            res_a += (*(img2))* c_L[hlen-3];
            img3 += Nc;
            res_d += (*(img3))* c_H[hlen-4];
            img4 -= Nc;
            res_v += (*(img4))* c_L[hlen-3];

            res_a += *(img1) * c_L[hlen-4];
            res_h += *(img2) * c_H[hlen-5];
            res_v += *(img3) * c_L[hlen-4];
            res_d += *(img4) * c_H[hlen-5];

            img1 += Nc;
            res_h += *(img1)* c_H[hlen-5];
            img2 -= Nc;
            res_a += (*(img2))* c_L[hlen-4];
            img3 += Nc;
            res_d += (*(img3))* c_H[hlen-5];
            img4 -= Nc;
            res_v += (*(img4))* c_L[hlen-4];

     
     
     
        img1 -= 4*Nc-1;
        img2 += 3*Nc+1;
        img3 -= 4*Nc-1;
        img4 += 3*Nc+1;

    

          c_a[row* Nc + col] = res_a;
          c_h[row* Nc + col] = res_h;
          c_v[row* Nc + col] = res_v;
          c_d[row* Nc + col] = res_d;
 
      }
   }


  for(int row=Nr2-filter_len;row<Nr2;++row)
  {
	  for(int col=0;col<Nc;++col)
	  {
   
          DTYPE res_a = 0, res_h = 0, res_v = 0, res_d = 0;

        // Convolution with periodic boundaries extension.
          for (int jy = 0; jy <= filter_len; jy++) {
              int idx_y = row*2 - c + jy;

              if (idx_y < 0) idx_y += (Nr + Nr_is_odd); // if N is odd, image is virtually extended
              // no "else if", since idx_y can be > N-1  after being incremented
              if (idx_y > Nr-1) {
                  if ((idx_y == Nr) && (Nr_is_odd)) idx_y--; // if N is odd, repeat the right-most element
                  else idx_y -= (Nr + Nr_is_odd); // if N is odd, image is virtually extended
              }

              res_a += tmp_a1[idx_y*Nc + col] * c_kern_L[hlen-1 - jy];
              res_h += tmp_a1[idx_y*Nc + col] * c_kern_H[hlen-1 - jy];
              res_v += tmp_a2[idx_y*Nc + col] * c_kern_L[hlen-1 - jy];
              res_d += tmp_a2[idx_y*Nc + col] * c_kern_H[hlen-1 - jy];
          }

          c_a[row* Nc + col] = res_a;
          c_h[row* Nc + col] = res_h;
          c_v[row* Nc + col] = res_v;
          c_d[row* Nc + col] = res_d;
 
      }
   }




}


void separable_wavelet_transform::forward_pass2_even_symmetric_unroll2(DTYPE* __restrict__ tmp_a1, DTYPE* __restrict__ tmp_a2, DTYPE* __restrict__ c_a, DTYPE* __restrict__ c_h, DTYPE* __restrict__ c_v, DTYPE* __restrict__ c_d, int Nr, int Nc, int hlen)
{
	
 //std::cout << "Nr: " << Nr<< " Nc: " << Nc << " hlen: "	<< hlen << std::endl;
 int Nr_is_odd = (Nr & 1);
 int Nr2 = (Nr + Nr_is_odd)/2;
 
 DTYPE*  c_L =  c_kern_L.data();
 DTYPE*  c_H =  c_kern_H.data();

 int c, hL, hR, filter_len;
  
  c = hlen/2 - 1;
  hL = c;
  hR = c+1;
  filter_len = hL+hR; 


  for(int row=0;row<filter_len;++row)
  {
	  for(int col=0;col<Nc;++col)
	  {
   
          DTYPE res_a = 0, res_h = 0, res_v = 0, res_d = 0;

        // Convolution with periodic boundaries extension.
          for (int jy = 0; jy <= filter_len; jy++) {
              int idx_y = row*2 - c + jy;

              if (idx_y < 0) idx_y += (Nr + Nr_is_odd); // if N is odd, image is virtually extended
              // no "else if", since idx_y can be > N-1  after being incremented
              if (idx_y > Nr-1) {
                  if ((idx_y == Nr) && (Nr_is_odd)) idx_y--; // if N is odd, repeat the right-most element
                  else idx_y -= (Nr + Nr_is_odd); // if N is odd, image is virtually extended
              }

              res_a += tmp_a1[idx_y*Nc + col] * c_kern_L[hlen-1 - jy];
              res_h += tmp_a1[idx_y*Nc + col] * c_kern_H[hlen-1 - jy];
              res_v += tmp_a2[idx_y*Nc + col] * c_kern_L[hlen-1 - jy];
              res_d += tmp_a2[idx_y*Nc + col] * c_kern_H[hlen-1 - jy];
          }

          c_a[row* Nc + col] = res_a;
          c_h[row* Nc + col] = res_h;
          c_v[row* Nc + col] = res_v;
          c_d[row* Nc + col] = res_d;
 
      }
   }

 
  DTYPE res_a = 0, res_h = 0, res_v = 0, res_d = 0;
  DTYPE *img1,*img2,*img3,*img4;
  
  #pragma omp parallel for private(res_a,res_h,res_v,res_d,img1,img2,img3,img4) schedule(static)
  for(int row=filter_len;row<Nr2-filter_len;++row)
  {
	  for(int col=0;col<Nc;++col)
	  {
 
        // Convolution with periodic boundaries extension.
          //#pragma omp simd aligned(tmp_a1,tmp_a2,c_kern_L,c_kern_H:16)
          //#pragma omp simd aligned(tmp_a1,tmp_a2,c_L,c_H:16)

         img1 = &tmp_a1[(2*row-c)*Nc+col];
         img2 = &tmp_a1[(2*row+c)*Nc+col];
         img3 = &tmp_a2[(2*row-c)*Nc+col];
         img4 = &tmp_a2[(2*row+c)*Nc+col];
         res_a = *(img2+Nc)*c_L[0]+ *(img1+c*Nc)*c_L[hR];
         res_h = *(img1)*c_H[0]+ *(img1+(c+1)*Nc)*c_H[hR-1];
         res_v = *(img4+Nc)*c_L[0]+ *(img3+c*Nc)*c_L[hR];
         res_d = *(img3)*c_H[0]+ *(img3+(c+1)*Nc)*c_H[hR-1];
 
        // Convolution with periodic boundaries extension.
  

            res_a += (*(img1)+ *(img2))* c_L[hlen-1];
            res_h += (*(img1+Nc)+ *(img2+Nc))* c_H[hlen-2];
            res_v += (*(img3)+ *(img4))* c_L[hlen-1];
            res_d += (*(img3+Nc)+ *(img4+Nc))* c_H[hlen-2];

           img1 += Nc;
           img2 -= Nc;
           img3 += Nc;
           img4 -= Nc;
 
            res_a += (*(img1)+ *(img2))* c_L[hlen-2];
            res_h += (*(img1+Nc)+ *(img2+Nc))* c_H[hlen-3];
            res_v += (*(img3)+ *(img4))* c_L[hlen-2];
            res_d += (*(img3+Nc)+ *(img4+Nc))* c_H[hlen-3];


       /*
          res_a=0;
          res_h = 0;
          res_v = 0;
          res_d = 0; 
          for (int jy = 0; jy <= filter_len; jy++) {
              int idx_y = row*2 - c + jy;


              res_a += tmp_a1[idx_y*Nc + col] * c_L[hlen-1 - jy];
              res_h += tmp_a1[idx_y*Nc + col] * c_H[hlen-1 - jy];
              res_v += tmp_a2[idx_y*Nc + col] * c_L[hlen-1 - jy];
              res_d += tmp_a2[idx_y*Nc + col] * c_H[hlen-1 - jy];
       
          }*/

          c_a[row* Nc + col] = res_a;
          c_h[row* Nc + col] = res_h;
          c_v[row* Nc + col] = res_v;
          c_d[row* Nc + col] = res_d;
 
      }
   }


  for(int row=Nr2-filter_len;row<Nr2;++row)
  {
	  for(int col=0;col<Nc;++col)
	  {
   
          DTYPE res_a = 0, res_h = 0, res_v = 0, res_d = 0;

        // Convolution with periodic boundaries extension.
          for (int jy = 0; jy <= filter_len; jy++) {
              int idx_y = row*2 - c + jy;

              if (idx_y < 0) idx_y += (Nr + Nr_is_odd); // if N is odd, image is virtually extended
              // no "else if", since idx_y can be > N-1  after being incremented
              if (idx_y > Nr-1) {
                  if ((idx_y == Nr) && (Nr_is_odd)) idx_y--; // if N is odd, repeat the right-most element
                  else idx_y -= (Nr + Nr_is_odd); // if N is odd, image is virtually extended
              }

              res_a += tmp_a1[idx_y*Nc + col] * c_kern_L[hlen-1 - jy];
              res_h += tmp_a1[idx_y*Nc + col] * c_kern_H[hlen-1 - jy];
              res_v += tmp_a2[idx_y*Nc + col] * c_kern_L[hlen-1 - jy];
              res_d += tmp_a2[idx_y*Nc + col] * c_kern_H[hlen-1 - jy];
          }

          c_a[row* Nc + col] = res_a;
          c_h[row* Nc + col] = res_h;
          c_v[row* Nc + col] = res_v;
          c_d[row* Nc + col] = res_d;
 
      }
   }



}


void separable_wavelet_transform::forward_pass2_even_symmetric_unroll4(DTYPE* __restrict__ tmp_a1, DTYPE* __restrict__ tmp_a2, DTYPE* __restrict__ c_a, DTYPE* __restrict__ c_h, DTYPE* __restrict__ c_v, DTYPE* __restrict__ c_d, int Nr, int Nc, int hlen)
{
	
 //std::cout << "Nr: " << Nr<< " Nc: " << Nc << " hlen: "	<< hlen << std::endl;
 int Nr_is_odd = (Nr & 1);
 int Nr2 = (Nr + Nr_is_odd)/2;
 
   std::array<DTYPE,10> filter1;
    std::array<DTYPE,10> filter2;
 
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

 
 
 //DTYPE*  c_L =  c_kern_L.data();
// DTYPE*  c_H =  c_kern_H.data();
   DTYPE*  c_L =  filter1.data();
   DTYPE*  c_H =  filter2.data();

 int c, hR, hL,filter_len;
  
  c = hlen/2 - 1;
  hL = c;
  hR = c+1;
  
  filter_len = hL+hR; 


  for(int row=0;row<filter_len;++row)
  {
	  for(int col=0;col<Nc;++col)
	  {
   
          DTYPE res_a = 0, res_h = 0, res_v = 0, res_d = 0;

        // Convolution with periodic boundaries extension.
          for (int jy = 0; jy <= filter_len; jy++) {
              int idx_y = row*2 - c + jy;

              if (idx_y < 0) idx_y += (Nr + Nr_is_odd); // if N is odd, image is virtually extended
              // no "else if", since idx_y can be > N-1  after being incremented
              if (idx_y > Nr-1) {
                  if ((idx_y == Nr) && (Nr_is_odd)) idx_y--; // if N is odd, repeat the right-most element
                  else idx_y -= (Nr + Nr_is_odd); // if N is odd, image is virtually extended
              }

              res_a += tmp_a1[idx_y*Nc + col] * c_kern_L[hlen-1 - jy];
              res_h += tmp_a1[idx_y*Nc + col] * c_kern_H[hlen-1 - jy];
              res_v += tmp_a2[idx_y*Nc + col] * c_kern_L[hlen-1 - jy];
              res_d += tmp_a2[idx_y*Nc + col] * c_kern_H[hlen-1 - jy];
          }

          c_a[row* Nc + col] = res_a;
          c_h[row* Nc + col] = res_h;
          c_v[row* Nc + col] = res_v;
          c_d[row* Nc + col] = res_d;
 
      }
   }

 
  DTYPE res_a = 0, res_h = 0, res_v = 0, res_d = 0;
  DTYPE *img1,*img3;
  
  #pragma omp parallel for private(res_a,res_h,res_v,res_d,img1,img3) schedule(static,128) 
  for(int row=filter_len;row<Nr2-filter_len;++row)
  {
	  for(int col=0;col<Nc;++col)
	  {
 
        // Convolution with periodic boundaries extension.
          //#pragma omp simd aligned(tmp_a1,tmp_a2,c_kern_L,c_kern_H:16)
          //#pragma omp simd aligned(tmp_a1,tmp_a2,c_L,c_H:16)

         img1 = &tmp_a1[(2*row-c)*Nc+col];
         //img2 = &tmp_a1[(2*row+c)*Nc+col];
         img3 = &tmp_a2[(2*row-c)*Nc+col];
         //img4 = &tmp_a2[(2*row+c)*Nc+col];
         res_a = *(img1+c*Nc)*c_L[hR];
         res_h = *(img1)*c_H[0]+ *(img1+(c+1)*Nc)*c_H[hR-1];
         res_v = *(img3+c*Nc)*c_L[hR];
         res_d = *(img3)*c_H[0]+ *(img3+(c+1)*Nc)*c_H[hR-1];
 
        // Convolution with periodic boundaries extension.
  /*

            res_a += (*(img1))* c_L[hlen-1];
            res_v += (*(img3))* c_L[hlen-1];

            img1 += Nc;
            img3 += Nc;

            res_h += (*(img1))* c_H[hlen-2];
            res_d += (*(img3))* c_H[hlen-2];

            res_a += (*(img1))* c_L[hlen-2];
            res_v += (*(img3))* c_L[hlen-2];


            img1 += Nc;
            img3 += Nc;

            res_h += (*(img1))* c_H[hlen-3];
            res_d += (*(img3))* c_H[hlen-3];

            res_a += (*(img1))* c_L[hlen-3];
            res_v += (*(img3))* c_L[hlen-3];

            img1 += Nc;
            img3 += Nc;

            res_h += (*(img1))* c_H[hlen-4];
            res_d += (*(img3))* c_H[hlen-4];
 
            res_a += (*(img1))* c_L[hlen-4];
            res_v += (*(img3))* c_L[hlen-4];

            res_h += (*(img1+Nc))* c_H[hlen-5];
            res_d += (*(img3+Nc))* c_H[hlen-5];


            res_a += *(img2+Nc)*c_L[0];
            res_v += *(img4+Nc)*c_L[0];
 

            res_a += (*(img2))* c_L[hlen-1];
            res_h += (*(img2+Nc))* c_H[hlen-2];
            res_v += (*(img4))* c_L[hlen-1];
            res_d += (*(img4+Nc))* c_H[hlen-2];

            img2 -= Nc;
            img4 -= Nc;


            res_a += (*(img2))* c_L[hlen-2];
            res_h += (*(img2+Nc))* c_H[hlen-3];
            res_v += (*(img4))* c_L[hlen-2];
            res_d += (*(img4+Nc))* c_H[hlen-3];

           img2 -= Nc;
           img4 -= Nc;

            res_a += (*(img2))* c_L[hlen-3];
            res_h += (*(img2+Nc))* c_H[hlen-4];
            res_v += (*(img4))* c_L[hlen-3];
            res_d += (*(img4+Nc))* c_H[hlen-4];

           img2 -= Nc;
           img4 -= Nc;


            res_a += (*(img2))* c_L[hlen-4];
            res_h += (*(img2+Nc))* c_H[hlen-5];
            res_v += (*(img4))* c_L[hlen-4];
            res_d += (*(img4+Nc))* c_H[hlen-5];*/

       
          res_a=0;
          res_h = 0;
          res_v = 0;
          res_d = 0; 
          for (int jy = 0; jy <= filter_len; jy++) {
              int idx_y = row*2 - c + jy;


              res_a += tmp_a1[idx_y*Nc + col] * c_L[hlen-1 - jy];
              res_h += tmp_a1[idx_y*Nc + col] * c_H[hlen-1 - jy];
              res_v += tmp_a2[idx_y*Nc + col] * c_L[hlen-1 - jy];
              res_d += tmp_a2[idx_y*Nc + col] * c_H[hlen-1 - jy];
       
          }

          c_a[row* Nc + col] = res_a;
          c_h[row* Nc + col] = res_h;
          c_v[row* Nc + col] = res_v;
          c_d[row* Nc + col] = res_d;
 
      }
   }


  for(int row=Nr2-filter_len;row<Nr2;++row)
  {
	  for(int col=0;col<Nc;++col)
	  {
   
          DTYPE res_a = 0, res_h = 0, res_v = 0, res_d = 0;

        // Convolution with periodic boundaries extension.
          for (int jy = 0; jy <= filter_len; jy++) {
              int idx_y = row*2 - c + jy;

              if (idx_y < 0) idx_y += (Nr + Nr_is_odd); // if N is odd, image is virtually extended
              // no "else if", since idx_y can be > N-1  after being incremented
              if (idx_y > Nr-1) {
                  if ((idx_y == Nr) && (Nr_is_odd)) idx_y--; // if N is odd, repeat the right-most element
                  else idx_y -= (Nr + Nr_is_odd); // if N is odd, image is virtually extended
              }

              res_a += tmp_a1[idx_y*Nc + col] * c_kern_L[hlen-1 - jy];
              res_h += tmp_a1[idx_y*Nc + col] * c_kern_H[hlen-1 - jy];
              res_v += tmp_a2[idx_y*Nc + col] * c_kern_L[hlen-1 - jy];
              res_d += tmp_a2[idx_y*Nc + col] * c_kern_H[hlen-1 - jy];
          }

          c_a[row* Nc + col] = res_a;
          c_h[row* Nc + col] = res_h;
          c_v[row* Nc + col] = res_v;
          c_d[row* Nc + col] = res_d;
 
      }
   }



}


void separable_wavelet_transform::inverse_pass1(DTYPE* c_a, DTYPE* c_h, DTYPE* c_v, DTYPE* c_d, DTYPE* tmp1, DTYPE* tmp2, int Nr, int Nc, int Nr2, int hlen)
{


  int c, hL, hR,filter_len;
  int hlen2 = hlen/2; // Convolutions with even/odd indices of the kernels
  bool even = ((hlen2 & 1) == 0);
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
	  // TODO : more elegant
  }

  filter_len = hL+hR; 
  DTYPE*  c_IL =  c_kern_IL.data();
  DTYPE*  c_IH =  c_kern_IH.data();


  for(int row=0;row<filter_len;++row)
  {
	   int internal_row = row;
	   if(even) internal_row += 1; 
       int jy1 = c - internal_row/2;
       int jy2 = Nr - 1 - internal_row/2 + c;
       int offset_y = 1-(internal_row & 1);

	  for(int col=0;col<Nc;++col)
	  {

          DTYPE res_a = 0, res_h = 0, res_v = 0, res_d = 0;
          for (int jy = 0; jy <= filter_len; jy++) {
              int idx_y = internal_row/2 - c + jy;
              if (jy < jy1) idx_y += Nr;
              if (jy > jy2) idx_y -= Nr;

              res_a += c_a[idx_y*Nc + col] * c_kern_IL[hlen-1 - (2*jy + offset_y)];
              res_h += c_h[idx_y*Nc + col] * c_kern_IH[hlen-1 - (2*jy + offset_y)];
              res_v += c_v[idx_y*Nc + col] * c_kern_IL[hlen-1 - (2*jy + offset_y)];
              res_d += c_d[idx_y*Nc + col] * c_kern_IH[hlen-1 - (2*jy + offset_y)];
          }
          if ((hlen2 & 1) == 1) {
              tmp1[internal_row * Nc + col] = res_a + res_h;
              tmp2[internal_row * Nc + col] = res_v + res_d;
          }
          else {
            tmp1[(internal_row-1) * Nc + col] = res_a + res_h;
            tmp2[(internal_row-1) * Nc + col] = res_v + res_d;
         }
	  }
  }

   
  #pragma omp parallel for 
  for(int row=filter_len;row<Nr2-filter_len;++row)
  {
	   int internal_row = row;
	   if(even) internal_row += 1; 
       int offset_y = 1-(internal_row & 1);

	  for(int col=0;col<Nc;++col)
	  {

          DTYPE res_a = 0, res_h = 0, res_v = 0, res_d = 0;
         // #pragma omp simd aligned(c_a,c_h,c_v,c_d,c_IL,c_IH:16)
          for (int jy = 0; jy <= filter_len; jy++) {
              int idx_y = internal_row/2 - c + jy;

              res_a += c_a[idx_y*Nc + col] * c_IL[hlen-1 - (2*jy + offset_y)];
              res_h += c_h[idx_y*Nc + col] * c_IH[hlen-1 - (2*jy + offset_y)];
              res_v += c_v[idx_y*Nc + col] * c_IL[hlen-1 - (2*jy + offset_y)];
              res_d += c_d[idx_y*Nc + col] * c_IH[hlen-1 - (2*jy + offset_y)];
          }
          if ((hlen2 & 1) == 1) {
              tmp1[internal_row * Nc + col] = res_a + res_h;
              tmp2[internal_row * Nc + col] = res_v + res_d;
          }
          else {
            tmp1[(internal_row-1) * Nc + col] = res_a + res_h;
            tmp2[(internal_row-1) * Nc + col] = res_v + res_d;
         }
	  }
  }

  for(int row=Nr2-filter_len;row<Nr2;++row)
  {
	   int internal_row = row;
	   if(even) internal_row += 1; 
       int jy1 = c - internal_row/2;
       int jy2 = Nr - 1 - internal_row/2 + c;
       int offset_y = 1-(internal_row & 1);

	  for(int col=0;col<Nc;++col)
	  {

          DTYPE res_a = 0, res_h = 0, res_v = 0, res_d = 0;
          for (int jy = 0; jy <= filter_len; jy++) {
              int idx_y = internal_row/2 - c + jy;
              if (jy < jy1) idx_y += Nr;
              if (jy > jy2) idx_y -= Nr;

              res_a += c_a[idx_y*Nc + col] * c_kern_IL[hlen-1 - (2*jy + offset_y)];
              res_h += c_h[idx_y*Nc + col] * c_kern_IH[hlen-1 - (2*jy + offset_y)];
              res_v += c_v[idx_y*Nc + col] * c_kern_IL[hlen-1 - (2*jy + offset_y)];
              res_d += c_d[idx_y*Nc + col] * c_kern_IH[hlen-1 - (2*jy + offset_y)];
          }
          if ((hlen2 & 1) == 1) {
              tmp1[internal_row * Nc + col] = res_a + res_h;
              tmp2[internal_row * Nc + col] = res_v + res_d;
          }
          else {
            tmp1[(internal_row-1) * Nc + col] = res_a + res_h;
            tmp2[(internal_row-1) * Nc + col] = res_v + res_d;
         }
	  }
  }
  
}

void separable_wavelet_transform::inverse_pass1_even_symmetric_cdf_53_wavelet(const DTYPE* __restrict__ c_a, const DTYPE* __restrict__ c_h, const DTYPE* __restrict__ c_v, const DTYPE* __restrict__ c_d, DTYPE* __restrict__ tmp1, DTYPE* __restrict__ tmp2, int Nr, int Nc, int Nr2, int hlen)
{

  const int c = 1;

  const int filter_len = 2;
  
  const aligned_data_vec_t ifilter1={0.0,0.5,1.,0.5,0.,0.}; 
  
  const aligned_data_vec_t ifilter2={0.,-1.0/8.,-2.0/8.,6.0/8.,-2.0/8.,-1.0/8};
  
  


  for(int row=0;row<filter_len;++row)
  {
	   int internal_row = row;
       int jy1 = c - internal_row/2;
       int jy2 = Nr - 1 - internal_row/2 + c;
       int offset_y = 1-(internal_row & 1);

	  for(int col=0;col<Nc;++col)
	  {

          DTYPE res_a = 0, res_h = 0, res_v = 0, res_d = 0;
          for (int jy = 0; jy <= filter_len; jy++) {
              int idx_y = internal_row/2 - c + jy;
              if (jy < jy1) idx_y += Nr;
              if (jy > jy2) idx_y -= Nr;

              res_a += c_a[idx_y*Nc + col] * c_kern_IL[hlen-1 - (2*jy + offset_y)];
              res_h += c_h[idx_y*Nc + col] * c_kern_IH[hlen-1 - (2*jy + offset_y)];
              res_v += c_v[idx_y*Nc + col] * c_kern_IL[hlen-1 - (2*jy + offset_y)];
              res_d += c_d[idx_y*Nc + col] * c_kern_IH[hlen-1 - (2*jy + offset_y)];
          }
              tmp1[internal_row * Nc + col] = res_a + res_h;
              tmp2[internal_row * Nc + col] = res_v + res_d;
	  }
  }

   
  #pragma omp parallel for schedule(static)
  for(int row=filter_len;row<Nr2-filter_len;++row)
  {
	   int internal_row = row;
       int offset_y = 1-(internal_row & 1);
       int not_offset_y = 1-offset_y;

	  const DTYPE*  __restrict__ img1 ;
	  const DTYPE*  __restrict__ img2 ;
	  const DTYPE*  __restrict__ img1v ;
	  const DTYPE*  __restrict__ img2v;
	  const DTYPE*  __restrict__ img1h;
	  const DTYPE*  __restrict__ img2h;
	  const DTYPE*  __restrict__ img1d;
	  const DTYPE*  __restrict__ img2d;

	  const DTYPE*  __restrict__ p_ifilter1;
	  const DTYPE*  __restrict__ p_ifilter2;
	  
	  DTYPE res_a = 0.; 
	  DTYPE res_v = 0.; 
	  DTYPE res_h = 0.; 
	  DTYPE res_d = 0.; 
 
	  for(int col=0;col<Nc;++col)
	  {

          img1 = &c_a[(row/2-c)*Nc+col];
          img2 = &c_a[(row/2+c)*Nc+col];
          img1v = &c_v[(row/2-c)*Nc+col];
          img2v = &c_v[(row/2+c)*Nc+col];
          img1h = &c_h[(row/2-c)*Nc+col];
          img2h = &c_h[(row/2+c)*Nc+col];
          img1d = &c_d[(row/2-c)*Nc+col];
          img2d = &c_d[(row/2+c)*Nc+col];

          p_ifilter1 = &ifilter1[5];
          p_ifilter2 = &ifilter2[5];
          
          res_a = 0.; 
          res_v = 0.; 
          res_h = 0.; 
          res_d = 0.; 
          
          if(not_offset_y)
          {

	
            // res_a = *(img1) * *p_ifilter1;
            // res_v = *(img1v) * *p_ifilter1;
             res_h += (*(img1h)+*(img2h))* *p_ifilter2;
             res_d += (*(img1d)+*(img2d))* *p_ifilter2;
             
			 p_ifilter1-= 2;
			 p_ifilter2-= 2;

             img1 += Nc;
             img1v += Nc;

             img1h += Nc;
             img2h -= Nc;

             img1d += Nc;
             img2d -= Nc;

	
             res_a += (*(img1)+*(img2))* *p_ifilter1;
             res_v += (*(img1v)+*(img2v))* *p_ifilter1;

             res_h += *(img1h) * *p_ifilter2;
             res_d += *(img1d) * *p_ifilter2;
            
		  
		  }
          else
          {
			 p_ifilter1 -= 1;
			 p_ifilter2 -= 1;

	
			// res_h = *(img2h)* ifilter2[0];
			// res_d = *(img2d)* ifilter2[0];

             img2h -= Nc;
             img2d -= Nc;


             res_a += (*(img1)+*(img2))* *p_ifilter1;
             res_v += (*(img1v)+*(img2v))* *p_ifilter1;
             res_h += (*(img1h)+*(img2h))* *p_ifilter2;
             res_d += (*(img1d)+*(img2d))* *p_ifilter2;

             img1 += Nc;
             img2 -= Nc;
             img1v += Nc;
             img2v -= Nc;
             img1h += Nc;
             img2h -= Nc;
             img1d += Nc;
             img2d -= Nc;


             p_ifilter1-= 2;
             p_ifilter2-= 2;


			 res_a += *(img1)* *p_ifilter1;
			 res_v += *(img1v)* *p_ifilter1;
			 
		  }
 
              tmp1[internal_row * Nc + col] = res_a + res_h;
              tmp2[internal_row * Nc + col] = res_v + res_d;
	  }
  }

  for(int row=Nr2-filter_len;row<Nr2;++row)
  {
	   int internal_row = row;
       int jy1 = c - internal_row/2;
       int jy2 = Nr - 1 - internal_row/2 + c;
       int offset_y = 1-(internal_row & 1);

	  for(int col=0;col<Nc;++col)
	  {

          DTYPE res_a = 0, res_h = 0, res_v = 0, res_d = 0;
          for (int jy = 0; jy <= filter_len; jy++) {
              int idx_y = internal_row/2 - c + jy;
              if (jy < jy1) idx_y += Nr;
              if (jy > jy2) idx_y -= Nr;

              res_a += c_a[idx_y*Nc + col] * c_kern_IL[hlen-1 - (2*jy + offset_y)];
              res_h += c_h[idx_y*Nc + col] * c_kern_IH[hlen-1 - (2*jy + offset_y)];
              res_v += c_v[idx_y*Nc + col] * c_kern_IL[hlen-1 - (2*jy + offset_y)];
              res_d += c_d[idx_y*Nc + col] * c_kern_IH[hlen-1 - (2*jy + offset_y)];
          }
              tmp1[internal_row * Nc + col] = res_a + res_h;
              tmp2[internal_row * Nc + col] = res_v + res_d;
	  }
  }
 
 
}


void separable_wavelet_transform::inverse_pass1_even_symmetric_cdf_97_wavelet(const DTYPE* __restrict__ c_a, const DTYPE* __restrict__ c_h, const DTYPE* __restrict__ c_v, const DTYPE* __restrict__ c_d, DTYPE* __restrict__ tmp1, DTYPE* __restrict__ tmp2, int Nr, int Nc, int Nr2, int hlen)
{
 
  const int c = 2;

  const int filter_len = 4;
   
   const aligned_data_vec_t ifilter1={0.0,-0.091271763114 ,-0.057543526229,0.591271763114 ,1.11508705 ,0.591271763114,-0.057543526229 ,
                               -0.091271763114 ,0.0,0.0}; 
  
  const aligned_data_vec_t ifilter2={0.,0.026748757411,0.016864118443,-0.078223266529,-0.266864118443,
                                    0.602949018236,-0.266864118443,-0.078223266529,0.016864118443,0.026748757411 };
  
  for(int row=0;row<filter_len;++row)
  {
	   int internal_row = row;
       int jy1 = c - internal_row/2;
       int jy2 = Nr - 1 - internal_row/2 + c;
       int offset_y = 1-(internal_row & 1);

	  for(int col=0;col<Nc;++col)
	  {

          DTYPE res_a = 0, res_h = 0, res_v = 0, res_d = 0;
          for (int jy = 0; jy <= filter_len; jy++) {
              int idx_y = internal_row/2 - c + jy;
              if (jy < jy1) idx_y += Nr;
              if (jy > jy2) idx_y -= Nr;

              res_a += c_a[idx_y*Nc + col] * c_kern_IL[hlen-1 - (2*jy + offset_y)];
              res_h += c_h[idx_y*Nc + col] * c_kern_IH[hlen-1 - (2*jy + offset_y)];
              res_v += c_v[idx_y*Nc + col] * c_kern_IL[hlen-1 - (2*jy + offset_y)];
              res_d += c_d[idx_y*Nc + col] * c_kern_IH[hlen-1 - (2*jy + offset_y)];
          }
              tmp1[internal_row * Nc + col] = res_a + res_h;
              tmp2[internal_row * Nc + col] = res_v + res_d;
	  }
  }

   
  #pragma omp parallel for  schedule(static)
  for(int row=filter_len;row<Nr2-filter_len;++row)
  {
      
      int odd_row = (row & 1);

	  const DTYPE*  __restrict__ img1 = &c_a[(row/2-c)*Nc];
	  const DTYPE*  __restrict__ img2 = &c_a[(row/2+c)*Nc];
	  const DTYPE*  __restrict__ img1v = &c_v[(row/2-c)*Nc];
	  const DTYPE*  __restrict__ img2v = &c_v[(row/2+c)*Nc];
	  const DTYPE*  __restrict__ img1h = &c_h[(row/2-c)*Nc];
	  const DTYPE*  __restrict__ img2h = &c_h[(row/2+c)*Nc];
	  const DTYPE*  __restrict__ img1d = &c_d[(row/2-c)*Nc];
	  const DTYPE*  __restrict__ img2d = &c_d[(row/2+c)*Nc];

	  const DTYPE*  __restrict__ p_ifilter1;
	  const DTYPE*  __restrict__ p_ifilter2;
	  
	  DTYPE res_a = 0.; 
	  DTYPE res_v = 0.; 
	  DTYPE res_h = 0.; 
	  DTYPE res_d = 0.; 
 
	  for(int col=0;col<Nc;++col)
	  {
          
  

          p_ifilter1 = &ifilter1[9];
          p_ifilter2 = &ifilter2[9];
          
          res_a = 0.; 
          res_v = 0.; 
          res_h = 0.; 
          res_d = 0.; 
          
          if(odd_row)
          {

	
             //res_a = *(img1) * *p_ifilter1;
             //res_v = *(img1v) * *p_ifilter1;
             res_h += (*(img1h)+*(img2h))* *p_ifilter2;
             res_d += (*(img1d)+*(img2d))* *p_ifilter2;
             
			 p_ifilter1-= 2;
			 p_ifilter2-= 2;

             img1 += Nc;
             img1v += Nc;

             img1h += Nc;
             img2h -= Nc;

             img1d += Nc;
             img2d -= Nc;

	

             res_a += (*(img1)+*(img2))* *p_ifilter1;
             res_v += (*(img1v)+*(img2v))* *p_ifilter1;
             res_h += (*(img1h)+*(img2h))* *p_ifilter2;
             res_d += (*(img1d)+*(img2d))* *p_ifilter2;
             
             img1 += Nc;
             img2 -= Nc;
             
             
             
             img1v += Nc;
             img2v -= Nc;

             img1h += Nc;
             img1d += Nc;

             p_ifilter1-= 2;
             p_ifilter2-= 2;

	
             res_a += (*(img1)+*(img2))* *p_ifilter1;
             res_v += (*(img1v)+*(img2v))* *p_ifilter1;

             res_h += *(img1h) * *p_ifilter2;
             res_d += *(img1d) * *p_ifilter2;
             
             img1 -= 2*Nc-1;
             img2 += Nc+1;

             img1v -= 2*Nc-1;
             img2v += Nc+1;
             
             img1h -= 2*Nc-1;
             img2h += Nc+1;

             img1d -= 2*Nc-1;
             img2d += Nc+1;
 
		  
		  }
          else
          {
			 p_ifilter1 -= 1;
			 p_ifilter2 -= 1;

	
			// res_h = *(img2h)* ifilter2[0];
			// res_d = *(img2d)* ifilter2[0];

             img2h -= Nc;
             img2d -= Nc;


             res_a += (*(img1)+*(img2))* *p_ifilter1;
             res_v += (*(img1v)+*(img2v))* *p_ifilter1;
             res_h += (*(img1h)+*(img2h))* *p_ifilter2;
             res_d += (*(img1d)+*(img2d))* *p_ifilter2;

             img1 += Nc;
             img2 -= Nc;
             img1v += Nc;
             img2v -= Nc;
             img1h += Nc;
             img2h -= Nc;
             img1d += Nc;
             img2d -= Nc;


             p_ifilter1-= 2;
             p_ifilter2-= 2;


             res_a += (*(img1)+*(img2))* *p_ifilter1;
             res_v += (*(img1v)+*(img2v))* *p_ifilter1;
             res_h += (*(img1h)+*(img2h))* *p_ifilter2;
             res_d += (*(img1d)+*(img2d))* *p_ifilter2;


             img1 += Nc;
             img1v += Nc;

             p_ifilter1-= 2;


			 res_a += *(img1)* *p_ifilter1;
			 res_v += *(img1v)* *p_ifilter1;

             img1 -= 2*Nc-1;
             img2 += Nc+1;

             img1v -= 2*Nc-1;
             img2v += Nc+1;

             img1h -= Nc-1;
             img2h += 2*Nc+1;

             img1d -= Nc-1;
             img2d += 2*Nc+1;

			 
		  }

              tmp1[row * Nc + col] = res_a + res_h;
              tmp2[row * Nc + col] = res_v + res_d;
	  }
  }

  for(int row=Nr2-filter_len;row<Nr2;++row)
  {
	   int internal_row = row;
       int jy1 = c - internal_row/2;
       int jy2 = Nr - 1 - internal_row/2 + c;
       int offset_y = 1-(internal_row & 1);

	  for(int col=0;col<Nc;++col)
	  {

          DTYPE res_a = 0, res_h = 0, res_v = 0, res_d = 0;
          for (int jy = 0; jy <= filter_len; jy++) {
              int idx_y = internal_row/2 - c + jy;
              if (jy < jy1) idx_y += Nr;
              if (jy > jy2) idx_y -= Nr;

              res_a += c_a[idx_y*Nc + col] * c_kern_IL[hlen-1 - (2*jy + offset_y)];
              res_h += c_h[idx_y*Nc + col] * c_kern_IH[hlen-1 - (2*jy + offset_y)];
              res_v += c_v[idx_y*Nc + col] * c_kern_IL[hlen-1 - (2*jy + offset_y)];
              res_d += c_d[idx_y*Nc + col] * c_kern_IH[hlen-1 - (2*jy + offset_y)];
          }
              tmp1[internal_row * Nc + col] = res_a + res_h;
              tmp2[internal_row * Nc + col] = res_v + res_d;
	  }
  }
 
 
}



void separable_wavelet_transform::inverse_pass2(DTYPE* tmp1, DTYPE* tmp2, DTYPE* img, int Nr, int Nc, int Nc2, int hlen)
{


  int c, hL, hR,filter_len;
  int hlen2 = hlen/2; // Convolutions with even/odd indices of the kernels
  bool even = ((hlen2 & 1) == 0);
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
	  // TODO : more elegant
  }

  filter_len = hL+hR; 

  DTYPE*  c_IL =  c_kern_IL.data();
  DTYPE*  c_IH =  c_kern_IH.data();

  #pragma omp parallel for  
  for(int row=0;row<Nr;++row)
  {

	  for(int col=0;col<filter_len;++col)
	  {
  	        int internal_col = col;
  	        if(even) internal_col += 1; 

	        int jx1 = c - internal_col/2;
	        int jx2 = Nc - 1 - internal_col/2 + c;
	        int offset_x = 1-(internal_col & 1);
	
	        DTYPE res_1 = 0, res_2 = 0;
	        for (int jx = 0; jx <= filter_len; jx++) {
	            int idx_x = internal_col/2 - c + jx;
	            if (jx < jx1) idx_x += Nc;
	            if (jx > jx2) idx_x -= Nc;
	
	            res_1 += tmp1[row*Nc + idx_x] * c_kern_IL[hlen-1 - (2*jx + offset_x)];
	            res_2 += tmp2[row*Nc + idx_x] * c_kern_IH[hlen-1 - (2*jx + offset_x)];
	        }
	        if ((hlen2 & 1) == 1) img[row * Nc2 + internal_col] = res_1 + res_2;
	        else img[row * Nc2 + (internal_col-1)] = res_1 + res_2;

	  }


	  for(int col=filter_len;col<Nc2-filter_len;++col)
	  {
  	        int internal_col = col;
  	        if(even) internal_col += 1; 

	        //int jx1 = c - col/2;
	        //int jx2 = Nc - 1 - col/2 + c;
	        int offset_x = 1-(internal_col & 1);
	
	        DTYPE res_1 = 0, res_2 = 0;
	       // #pragma omp simd aligned(tmp1,tmp2,c_kern_IL,c_kern_IH:16)
	       // #pragma omp simd aligned(c_IL,c_IH:16)
	        for (int jx = 0; jx <= filter_len; jx++) {
	            int idx_x = internal_col/2 - c + jx;
	           // if (jx < jx1) idx_x += Nc;
	           // if (jx > jx2) idx_x -= Nc;
	
	            res_1 += tmp1[row*Nc + idx_x] * c_IL[hlen-1 - (2*jx + offset_x)];
	            res_2 += tmp2[row*Nc + idx_x] * c_IH[hlen-1 - (2*jx + offset_x)];
	        }
	        if ((hlen2 & 1) == 1) img[row * Nc2 + internal_col] = res_1 + res_2;
	        else img[row * Nc2 + (internal_col-1)] = res_1 + res_2;

	  }

	  for(int col=Nc2-filter_len;col<Nc2;++col)
	  {
  	        int internal_col = col;
  	        if(even) internal_col += 1; 

	        int jx1 = c - internal_col/2;
	        int jx2 = Nc - 1 - internal_col/2 + c;
	        int offset_x = 1-(internal_col & 1);
	
	        DTYPE res_1 = 0, res_2 = 0;
	        for (int jx = 0; jx <= filter_len; jx++) {
	            int idx_x = internal_col/2 - c + jx;
	            if (jx < jx1) idx_x += Nc;
	            if (jx > jx2) idx_x -= Nc;
	
	            res_1 += tmp1[row*Nc + idx_x] * c_kern_IL[hlen-1 - (2*jx + offset_x)];
	            res_2 += tmp2[row*Nc + idx_x] * c_kern_IH[hlen-1 - (2*jx + offset_x)];
	        }
	        if ((hlen2 & 1) == 1) img[row * Nc2 + internal_col] = res_1 + res_2;
	        else img[row * Nc2 + (internal_col-1)] = res_1 + res_2;

	  }


  }

 
}

void separable_wavelet_transform::inverse_pass2_even_symmetric_cdf_53_wavelet(const DTYPE* __restrict__ tmp1, const DTYPE* __restrict__ tmp2, DTYPE* __restrict__ img, int Nr, int Nc, int Nc2, int hlen)
{

  const int c = 1;

   const int filter_len = 2;
   
 
  
  const aligned_data_vec_t ifilter1={0.0,0.5,1.,0.5,0.,0.}; 
  
  const aligned_data_vec_t ifilter2={0.,-1.0/8.,-2.0/8.,6.0/8.,-2.0/8.,-1.0/8};
  
  

  const DTYPE*  c_IL =  ifilter1.data();
  const DTYPE*  c_IH =  ifilter2.data();

  
  #pragma omp parallel for  schedule(static)
  for(int row=0;row<Nr;++row)
  {

	  for(int col=0;col<filter_len;++col)
	  {
  	        int internal_col = col;

	        int jx1 = c - internal_col/2;
	        int jx2 = Nc - 1 - internal_col/2 + c;
	        int offset_x = 1-(internal_col & 1);
	
	        DTYPE res_1 = 0, res_2 = 0;
	        for (int jx = 0; jx <= filter_len; jx++) {
	            int idx_x = internal_col/2 - c + jx;
	            if (jx < jx1) idx_x += Nc;
	            if (jx > jx2) idx_x -= Nc;
	
	            res_1 += tmp1[row*Nc + idx_x] * c_kern_IL[hlen-1 - (2*jx + offset_x)];
	            res_2 += tmp2[row*Nc + idx_x] * c_kern_IH[hlen-1 - (2*jx + offset_x)];
	        }
	        img[row * Nc2 + internal_col] = res_1 + res_2;
	       

	  }


	  DTYPE res_1 = 0, res_2 = 0;
	  const DTYPE* __restrict__ tmp1_1;
	  const DTYPE* __restrict__ tmp1_2;
	  const DTYPE* __restrict__ filter_il;

	  const DTYPE* __restrict__ tmp2_1;
	  const DTYPE* __restrict__ tmp2_2;
	  const DTYPE* __restrict__ filter_ih;

	  for(int col=filter_len;col<Nc2-filter_len;++col)
	  {
  	        int internal_col = col;

	        int offset_x = 1-(internal_col & 1);
	
	        res_1 = 0.;
	        res_2 = 0.;
	        tmp1_1 = &tmp1[row*Nc + col/2-c];
	        tmp1_2 = &tmp1[row*Nc + col/2+c];
	        filter_il = &c_IL[5];

	        tmp2_1 = &tmp2[row*Nc + col/2-c];
	        tmp2_2 = &tmp2[row*Nc + col/2+c];
	        filter_ih = &c_IH[5];

	  
	        
	        if(offset_x)
	        {
				//res_2 = *tmp2_2 * c_IH[0];
				--filter_il;
				--filter_ih;
				--tmp2_2;
				
				res_1  = ((*tmp1_1)+(*tmp1_2))* *filter_il;
				res_2  += ((*tmp2_1)+(*tmp2_2))* *filter_ih;
				
			

				++tmp1_1;
				filter_il -= 2;
				res_1  += ((*tmp1_1))* *filter_il;
				
			}
			else
			{
				res_2  = ((*tmp2_1)+(*tmp2_2))* *filter_ih;
				//res_1  = ((*tmp1_1))* *filter_il;
				++tmp1_1;
				++tmp2_1;
				--tmp2_2;
				
				filter_il -= 2;
				filter_ih -= 2;
			
				res_1  += ((*tmp1_1)+(*tmp1_2))* *filter_il;
				res_2  += ((*tmp2_1))* *filter_ih;
			
			}
			
	   
	        img[row * Nc2 + internal_col] = res_1 + res_2;
	        

	  }

	  for(int col=Nc2-filter_len;col<Nc2;++col)
	  {
  	        int internal_col = col;

	        int jx1 = c - internal_col/2;
	        int jx2 = Nc - 1 - internal_col/2 + c;
	        int offset_x = 1-(internal_col & 1);
	
	        DTYPE res_1 = 0, res_2 = 0;
	        for (int jx = 0; jx <= filter_len; jx++) {
	            int idx_x = internal_col/2 - c + jx;
	            if (jx < jx1) idx_x += Nc;
	            if (jx > jx2) idx_x -= Nc;
	
	            res_1 += tmp1[row*Nc + idx_x] * c_kern_IL[hlen-1 - (2*jx + offset_x)];
	            res_2 += tmp2[row*Nc + idx_x] * c_kern_IH[hlen-1 - (2*jx + offset_x)];
	        }
	        img[row * Nc2 + internal_col] = res_1 + res_2;
	        

	  }


  }


}

void separable_wavelet_transform::inverse_pass2_even_symmetric_cdf_97_wavelet(const DTYPE* __restrict__ tmp1, const DTYPE* __restrict__ tmp2, DTYPE* __restrict__ img, int Nr, int Nc, int Nc2, int hlen)
{

   const int c = 2;

   const int filter_len = 4;
   
   const aligned_data_vec_t ifilter1={0.0,-0.091271763114 ,-0.057543526229,0.591271763114 ,1.11508705 ,0.591271763114,-0.057543526229 ,
                               -0.091271763114 ,0.0,0.0}; 
  
   const aligned_data_vec_t ifilter2={0.,0.026748757411,0.016864118443,-0.078223266529,-0.266864118443,
                                    0.602949018236,-0.266864118443,-0.078223266529,0.016864118443,0.026748757411 };

  const DTYPE*  c_IL =  ifilter1.data();
  const DTYPE*  c_IH =  ifilter2.data();

 
  
  #pragma omp parallel for schedule(static)
  for(int row=0;row<Nr;++row)
  {

	  for(int col=0;col<filter_len;++col)
	  {
  	        int internal_col = col;

	        int jx1 = c - internal_col/2;
	        int jx2 = Nc - 1 - internal_col/2 + c;
	        int offset_x = 1-(internal_col & 1);
	
	        DTYPE res_1 = 0, res_2 = 0;
	        for (int jx = 0; jx <= filter_len; jx++) {
	            int idx_x = internal_col/2 - c + jx;
	            if (jx < jx1) idx_x += Nc;
	            if (jx > jx2) idx_x -= Nc;
	
	            res_1 += tmp1[row*Nc + idx_x] * c_kern_IL[hlen-1 - (2*jx + offset_x)];
	            res_2 += tmp2[row*Nc + idx_x] * c_kern_IH[hlen-1 - (2*jx + offset_x)];
	        }
	        img[row * Nc2 + internal_col] = res_1 + res_2;
	       

	  }


	  DTYPE res_1 = 0, res_2 = 0;
	  const DTYPE* __restrict__ tmp1_1;
	  const DTYPE* __restrict__ tmp1_2;
	  const DTYPE* __restrict__ filter_il;

	  const DTYPE* __restrict__ tmp2_1;
	  const DTYPE* __restrict__ tmp2_2;
	  const DTYPE* __restrict__ filter_ih;

	  for(int col=filter_len;col<Nc2-filter_len;++col)
	  {
 	        bool even_col = !(col & 1);
	
	        res_1 = 0.;
	        res_2 = 0.;
	        tmp1_1 = &tmp1[row*Nc + col/2-c];
	        tmp1_2 = &tmp1[row*Nc + col/2+c];
	        filter_il = &c_IL[9];

	        tmp2_1 = &tmp2[row*Nc + col/2-c];
	        tmp2_2 = &tmp2[row*Nc + col/2+c];
	        filter_ih = &c_IH[9];

	       
	        
	        if(even_col)
	        {
				//res_2 = *tmp2_2 * c_IH[0];
				--filter_il;
				--filter_ih;
				--tmp2_2;
				
				res_1  = ((*tmp1_1)+(*tmp1_2))* *filter_il;
				res_2  += ((*tmp2_1)+(*tmp2_2))* *filter_ih;
				
				++tmp1_1;
				--tmp1_2;
				++tmp2_1;
				--tmp2_2;
				
				filter_il -= 2;
				filter_ih -= 2;

				res_1  += ((*tmp1_1)+(*tmp1_2))* *filter_il;
				res_2  += ((*tmp2_1)+(*tmp2_2))* *filter_ih;

				++tmp1_1;
				filter_il -= 2;
				res_1  += ((*tmp1_1))* *filter_il;
				
			}
			else
			{
				res_2  = ((*tmp2_1)+(*tmp2_2))* *filter_ih;
				//res_1  = ((*tmp1_1))* *filter_il;
				++tmp1_1;
				++tmp2_1;
				--tmp2_2;

				filter_il -= 2;
				filter_ih -= 2;
				
				res_1  += ((*tmp1_1)+(*tmp1_2))* *filter_il;
				res_2  += ((*tmp2_1)+(*tmp2_2))* *filter_ih;

				++tmp1_1;
				--tmp1_2;
				++tmp2_1;
				filter_il -= 2;
				filter_ih -= 2;
				res_1  += ((*tmp1_1)+(*tmp1_2))* *filter_il;
				res_2  += ((*tmp2_1))* *filter_ih;
			
			}
			
	   
	        img[row * Nc2 + col] = res_1 + res_2;
	        

	  }

	  for(int col=Nc2-filter_len;col<Nc2;++col)
	  {
  	        int internal_col = col;

	        int jx1 = c - internal_col/2;
	        int jx2 = Nc - 1 - internal_col/2 + c;
	        int offset_x = 1-(internal_col & 1);
	
	        DTYPE res_1 = 0, res_2 = 0;
	        for (int jx = 0; jx <= filter_len; jx++) {
	            int idx_x = internal_col/2 - c + jx;
	            if (jx < jx1) idx_x += Nc;
	            if (jx > jx2) idx_x -= Nc;
	
	            res_1 += tmp1[row*Nc + idx_x] * c_kern_IL[hlen-1 - (2*jx + offset_x)];
	            res_2 += tmp2[row*Nc + idx_x] * c_kern_IH[hlen-1 - (2*jx + offset_x)];
	        }
	        img[row * Nc2 + internal_col] = res_1 + res_2;
	        

	  }


  }

}


void separable_wavelet_transform::forward_swt_pass1(DTYPE* img, DTYPE* tmp_a1, DTYPE* tmp_a2, int Nr, int Nc, int hlen, int level)
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
	        DTYPE res_tmp_a1 = 0, res_tmp_a2 = 0;
	        DTYPE img_val;
	
	        // Convolution with periodic boundaries extension.
	        // The filters are 2-upsampled at each level : [h0, h1, h2, h3] --> [h0, 0, h1, 0, h2, 0, h3, 0]
	       for (int jx = 0; jx <= hR+hL; jx++) {
	            int idx_x = col + jx*factor - c;
	            if (factor*jx < jx1) idx_x += Nc;
	            if (factor*jx > jx2) idx_x -= Nc;
	
	            img_val = img[(row)*Nc + idx_x];
	            res_tmp_a1 += img_val * c_kern_L[hlen-1 - jx];
	            res_tmp_a2 += img_val * c_kern_H[hlen-1 - jx];
	        }
	
	        tmp_a1[row* Nc + col] = res_tmp_a1;
	        tmp_a2[row* Nc + col] = res_tmp_a2;

	  }
  }
  
  
  
}

void separable_wavelet_transform::forward_swt_pass2(DTYPE* tmp_a1, DTYPE* tmp_a2, DTYPE* c_a, DTYPE* c_h, DTYPE* c_v, DTYPE* c_d, int Nr, int Nc, int hlen, int level)
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
	        int jy1 = c - row;
	        int jy2 = Nr - 1 - row + c;
	        DTYPE res_a = 0, res_h = 0, res_v = 0, res_d = 0;
	
	        // Convolution with periodic boundaries extension.
	        // The filters are 2-upsampled at each level : [h0, h1, h2, h3] --> [h0, 0, h1, 0, h2, 0, h3, 0]
	        for (int jy = 0; jy <= hR+hL; jy++) {
	            int idx_y = row + factor*jy - c;
	            if (factor*jy < jy1) idx_y += Nr;
	            if (factor*jy > jy2) idx_y -= Nr;
	
	            res_a += tmp_a1[idx_y*Nc + col] * c_kern_L[hlen-1 - jy];
	            res_h += tmp_a1[idx_y*Nc + col] * c_kern_H[hlen-1 - jy];
	            res_v += tmp_a2[idx_y*Nc + col] * c_kern_L[hlen-1 - jy];
	            res_d += tmp_a2[idx_y*Nc + col] * c_kern_H[hlen-1 - jy];
	        }
	
	        c_a[row* Nc + col] = res_a;
	        c_h[row* Nc + col] = res_h;
	        c_v[row* Nc + col] = res_v;
	        c_d[row* Nc + col] = res_d;

      }
  }	
}

void separable_wavelet_transform::inverse_swt_pass1(DTYPE* c_a, DTYPE* c_h, DTYPE* c_v, DTYPE* c_d, DTYPE* tmp1, DTYPE* tmp2, int Nr, int Nc, int hlen, int level)
{
	
  for(int row=0;row<Nr;++row)
  {
	  for(int col=0;col<Nc;++col)
	  {
	        int c, hL, hR;
	        int factor = 1 << (level - 1);
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
	        int offset_y = 1-(row & 1);
	        offset_y = 0;
	
	        DTYPE res_a = 0, res_h = 0, res_v = 0, res_d = 0;
	        for (int jy = 0; jy <= hR+hL; jy++) {
	            int idx_y = row - c + factor*jy;
	            if (factor*jy < jy1) idx_y += Nr;
	            if (factor*jy > jy2) idx_y -= Nr;
	
	            res_a += c_a[idx_y*Nc + col] * c_kern_IL[hlen-1 - (jy + offset_y)]/2;
	            res_h += c_h[idx_y*Nc + col] * c_kern_IH[hlen-1 - (jy + offset_y)]/2;
	            res_v += c_v[idx_y*Nc + col] * c_kern_IL[hlen-1 - (jy + offset_y)]/2;
	            res_d += c_d[idx_y*Nc + col] * c_kern_IH[hlen-1 - (jy + offset_y)]/2;
	        }
	        tmp1[row * Nc + col] = res_a + res_h;
	        tmp2[row * Nc + col] = res_v + res_d;
	  }
  }
}

void separable_wavelet_transform::inverse_swt_pass2(DTYPE* tmp1, DTYPE* tmp2, DTYPE* img, int Nr, int Nc, int hlen, int level)
{
	
  for(int row=0;row<Nr;++row)
  {
	  for(int col=0;col<Nc;++col)
	  {
	        int c, hL, hR;
	        int factor = 1 << (level - 1);
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
	        int jx1 = c - col;
	        int jx2 = Nc - 1 - col + c;
	        int offset_x = 1-(col & 1);
	        offset_x = 0;
	
	        DTYPE res_1 = 0, res_2 = 0;
	        for (int jx = 0; jx <= hR+hL; jx++) {
	            int idx_x = col - c + factor*jx;
	            if (factor*jx < jx1) idx_x += Nc;
	            if (factor*jx > jx2) idx_x -= Nc;
	
	            res_1 += tmp1[row*Nc + idx_x] * c_kern_IL[hlen-1 - (jx + offset_x)]/2;
	            res_2 += tmp2[row*Nc + idx_x] * c_kern_IH[hlen-1 - (jx + offset_x)]/2;
	        }
	        img[row * Nc + col] = res_1 + res_2;

	  }
  }
}


