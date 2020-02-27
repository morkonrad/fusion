#pragma once

//Precompilation parameter that determine whether we compute the DWT 53 or the DWT 97. Equal to 1, DWT 53. If 0, DWT 97.
#if !defined(DWT53_or_DWT97)
	#define DWT53_or_DWT97	0
#endif

//Data types employed to represent each input sample. DATATYPE_16BITS_or_32BITS equal to 1 indicates that int16_t are used. If its 0, normal int/float of 32 bites are used. 
#if !defined(DATATYPE_16BITS_or_32BITS)
	#define DATATYPE_16BITS_or_32BITS	1
#endif



//The following precompilation parameters determine the data types employed in the device, depending on the use of the 5/3 or the 9/7 DWT. 
//If we compute the DWT 53
#if DWT53_or_DWT97 == 1
	#define REG_DATATYPE			int
	//In the DWT 5/3 each Data block is overlapped with its neighbours 4 samples 
	#define OVERLAP					4
	#if DATATYPE_16BITS_or_32BITS == 1
		#define DATATYPE			int16_t
		#define DATATYPE2 			int	
	#else
		#define DATATYPE 			int
		#define DATATYPE2 			int2
	#endif

//If we compute the DWT 97
#else
	#define REG_DATATYPE			float
//In the DWT 9/7 each Data block is overlapped with its neighbours 8 samples 
	#define OVERLAP					8
	#if DATATYPE_16BITS_or_32BITS == 1
		#define DATATYPE 			int16_t
		#define DATATYPE2 			int
	#else
		#define DATATYPE 			float
		#define DATATYPE2 			float2
	#endif
#endif


