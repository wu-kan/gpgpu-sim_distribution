//===============================================================================================================================================================================================================
//===============================================================================================================================================================================================================
//	DEFINE / INCLUDE
//===============================================================================================================================================================================================================
//===============================================================================================================================================================================================================

//======================================================================================================================================================
//	LIBRARIES
//======================================================================================================================================================

#include <stdlib.h>
#include <math.h>
#include <string.h>

#include <cuda.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 *  avilib.h
 *
 *  Copyright (C) Thomas �streich - June 2001
 *  multiple audio track support Copyright (C) 2002 Thomas �streich
 *
 *  Original code:
 *  Copyright (C) 1999 Rainer Johanni <Rainer@Johanni.de> 
 *
 *  This file is part of transcode, a linux video stream processing tool
 *      
 *  transcode is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2, or (at your option)
 *  any later version.
 *   
 *  transcode is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *   
 *  You should have received a copy of the GNU General Public License
 *  along with GNU Make; see the file COPYING.  If not, write to
 *  the Free Software Foundation, 675 Mass Ave, Cambridge, MA 02139, USA. 
 *
 */

#include <sys/types.h>
#include <sys/stat.h>
#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>
// #include <windows.h>
#include <inttypes.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

#ifndef AVILIB_H
	#define AVILIB_H

	#define AVI_MAX_TRACKS 8

	typedef struct
	{
	  unsigned long key;
	  unsigned long pos;
	  unsigned long len;
	} video_index_entry;

	typedef struct
	{
	   unsigned long pos;
	   unsigned long len;
	   unsigned long tot;
	} audio_index_entry;

	typedef struct track_s
	{

		long   a_fmt;             /* Audio format, see #defines below */
		long   a_chans;           /* Audio channels, 0 for no audio */
		long   a_rate;            /* Rate in Hz */
		long   a_bits;            /* bits per audio sample */
		long   mp3rate;           /* mp3 bitrate kbs*/

		long   audio_strn;        /* Audio stream number */
		long   audio_bytes;       /* Total number of bytes of audio data */
		long   audio_chunks;      /* Chunks of audio data in the file */

		char   audio_tag[4];      /* Tag of audio data */
		long   audio_posc;        /* Audio position: chunk */
		long   audio_posb;        /* Audio position: byte within chunk */
	 
		long  a_codech_off;       /* absolut offset of audio codec information */ 
		long  a_codecf_off;       /* absolut offset of audio codec information */ 

		audio_index_entry *audio_index;

	} track_t;

	typedef struct
	{
	  
	  long   fdes;              /* File descriptor of AVI file */
	  long   mode;              /* 0 for reading, 1 for writing */
	  
	  long   width;             /* Width  of a video frame */
	  long   height;            /* Height of a video frame */
	  double fps;               /* Frames per second */
	  char   compressor[8];     /* Type of compressor, 4 bytes + padding for 0 byte */
	  char   compressor2[8];     /* Type of compressor, 4 bytes + padding for 0 byte */
	  long   video_strn;        /* Video stream number */
	  long   video_frames;      /* Number of video frames */
	  char   video_tag[4];      /* Tag of video data */
	  long   video_pos;         /* Number of next frame to be read
					   (if index present) */
	  
	  unsigned long max_len;    /* maximum video chunk present */
	  
	  track_t track[AVI_MAX_TRACKS];  // up to AVI_MAX_TRACKS audio tracks supported
	  
	  unsigned long pos;        /* position in file */
	  long   n_idx;             /* number of index entries actually filled */
	  long   max_idx;           /* number of index entries actually allocated */
	  
	  long  v_codech_off;       /* absolut offset of video codec (strh) info */ 
	  long  v_codecf_off;       /* absolut offset of video codec (strf) info */ 
	  
	  unsigned char (*idx)[16]; /* index entries (AVI idx1 tag) */
	  video_index_entry *video_index;
	  
	  unsigned long last_pos;          /* Position of last frame written */
	  unsigned long last_len;          /* Length of last frame written */
	  int must_use_index;              /* Flag if frames are duplicated */
	  unsigned long   movi_start;
	  
	  int anum;            // total number of audio tracks 
	  int aptr;            // current audio working track 
	  
	} avi_t;

	#define AVI_MODE_WRITE  0
	#define AVI_MODE_READ   1

	/* The error codes delivered by avi_open_input_file */

	#define AVI_ERR_SIZELIM      1     /* The write of the data would exceed
										  the maximum size of the AVI file.
										  This is more a warning than an error
										  since the file may be closed safely */

	#define AVI_ERR_OPEN         2     /* Error opening the AVI file - wrong path
										  name or file nor readable/writable */

	#define AVI_ERR_READ         3     /* Error reading from AVI File */

	#define AVI_ERR_WRITE        4     /* Error writing to AVI File,
										  disk full ??? */

	#define AVI_ERR_WRITE_INDEX  5     /* Could not write index to AVI file
										  during close, file may still be
										  usable */

	#define AVI_ERR_CLOSE        6     /* Could not write header to AVI file
										  or not truncate the file during close,
										  file is most probably corrupted */

	#define AVI_ERR_NOT_PERM     7     /* Operation not permitted:
										  trying to read from a file open
										  for writing or vice versa */

	#define AVI_ERR_NO_MEM       8     /* malloc failed */

	#define AVI_ERR_NO_AVI       9     /* Not an AVI file */

	#define AVI_ERR_NO_HDRL     10     /* AVI file has no has no header list,
										  corrupted ??? */

	#define AVI_ERR_NO_MOVI     11     /* AVI file has no has no MOVI list,
										  corrupted ??? */

	#define AVI_ERR_NO_VIDS     12     /* AVI file contains no video data */

	#define AVI_ERR_NO_IDX      13     /* The file has been opened with
										  getIndex==0, but an operation has been
										  performed that needs an index */

	/* Possible Audio formats */

	#ifndef WAVE_FORMAT_PCM
		#define WAVE_FORMAT_UNKNOWN             (0x0000)
		#define WAVE_FORMAT_PCM                 (0x0001)
		#define WAVE_FORMAT_ADPCM               (0x0002)
		#define WAVE_FORMAT_IBM_CVSD            (0x0005)
		#define WAVE_FORMAT_ALAW                (0x0006)
		#define WAVE_FORMAT_MULAW               (0x0007)
		#define WAVE_FORMAT_OKI_ADPCM           (0x0010)
		#define WAVE_FORMAT_DVI_ADPCM           (0x0011)
		#define WAVE_FORMAT_DIGISTD             (0x0015)
		#define WAVE_FORMAT_DIGIFIX             (0x0016)
		#define WAVE_FORMAT_YAMAHA_ADPCM        (0x0020)
		#define WAVE_FORMAT_DSP_TRUESPEECH      (0x0022)
		#define WAVE_FORMAT_GSM610              (0x0031)
		#define IBM_FORMAT_MULAW                (0x0101)
		#define IBM_FORMAT_ALAW                 (0x0102)
		#define IBM_FORMAT_ADPCM                (0x0103)
	#endif

	avi_t* AVI_open_output_file(char * filename);
	void AVI_set_video(avi_t *AVI, int width, int height, double fps, char *compressor);
	void AVI_set_audio(avi_t *AVI, int channels, long rate, int bits, int format, long mp3rate);
	int  AVI_write_frame(avi_t *AVI, char *data, long bytes, int keyframe);
	int  AVI_dup_frame(avi_t *AVI);
	int  AVI_write_audio(avi_t *AVI, char *data, long bytes);
	int  AVI_append_audio(avi_t *AVI, char *data, long bytes);
	long AVI_bytes_remain(avi_t *AVI);
	int  AVI_close(avi_t *AVI);
	long AVI_bytes_written(avi_t *AVI);

	avi_t *AVI_open_input_file(char *filename, int getIndex);
	avi_t *AVI_open_fd(int fd, int getIndex);
	int avi_parse_input_file(avi_t *AVI, int getIndex);
	long AVI_audio_mp3rate(avi_t *AVI);
	long AVI_video_frames(avi_t *AVI);
	int  AVI_video_width(avi_t *AVI);
	int  AVI_video_height(avi_t *AVI);
	double AVI_frame_rate(avi_t *AVI);
	char* AVI_video_compressor(avi_t *AVI);

	int  AVI_audio_channels(avi_t *AVI);
	int  AVI_audio_bits(avi_t *AVI);
	int  AVI_audio_format(avi_t *AVI);
	long AVI_audio_rate(avi_t *AVI);
	long AVI_audio_bytes(avi_t *AVI);
	long AVI_audio_chunks(avi_t *AVI);

	long AVI_max_video_chunk(avi_t *AVI);

	long AVI_frame_size(avi_t *AVI, long frame);
	long AVI_audio_size(avi_t *AVI, long frame);
	int  AVI_seek_start(avi_t *AVI);
	int  AVI_set_video_position(avi_t *AVI, long frame);
	long AVI_get_video_position(avi_t *AVI, long frame);
	long AVI_read_frame(avi_t *AVI, char *vidbuf, int *keyframe);

	int  AVI_set_audio_position(avi_t *AVI, long byte);
	int  AVI_set_audio_bitrate(avi_t *AVI, long bitrate);

	long AVI_read_audio(avi_t *AVI, char *audbuf, long bytes);

	long AVI_audio_codech_offset(avi_t *AVI);
	long AVI_audio_codecf_offset(avi_t *AVI);
	long AVI_video_codech_offset(avi_t *AVI);
	long AVI_video_codecf_offset(avi_t *AVI);

	int  AVI_read_data(avi_t *AVI, char *vidbuf, long max_vidbuf,
								   char *audbuf, long max_audbuf,
								   long *len);

	void AVI_print_error(char *str);
	char *AVI_strerror();
	char *AVI_syserror();

	int AVI_scan(char *name);
	int AVI_dump(char *name, int mode);

	char *AVI_codec2str(short cc);
	int AVI_file_check(char *import_file);

	void AVI_info(avi_t *avifile);
	uint64_t AVI_max_size();
	int avi_update_header(avi_t *AVI);

	int AVI_set_audio_track(avi_t *AVI, int track);
	int AVI_get_audio_track(avi_t *AVI);
	int AVI_audio_tracks(avi_t *AVI);

	struct riff_struct 
	{
	  unsigned char id[4];   /* RIFF */
	  unsigned long len;
	  unsigned char wave_id[4]; /* WAVE */
	};


	struct chunk_struct 
	{
		unsigned char id[4];
		unsigned long len;
	};

	struct common_struct 
	{
		unsigned short wFormatTag;
		unsigned short wChannels;
		unsigned long dwSamplesPerSec;
		unsigned long dwAvgBytesPerSec;
		unsigned short wBlockAlign;
		unsigned short wBitsPerSample;  /* Only for PCM */
	};

	struct wave_header 
	{
		struct riff_struct   riff;
		struct chunk_struct  format;
		struct common_struct common;
		struct chunk_struct  data;
	};



	struct AVIStreamHeader {
	  long  fccType;
	  long  fccHandler;
	  long  dwFlags;
	  long  dwPriority;
	  long  dwInitialFrames;
	  long  dwScale;
	  long  dwRate;
	  long  dwStart;
	  long  dwLength;
	  long  dwSuggestedBufferSize;
	  long  dwQuality;
	  long  dwSampleSize;
	};

#endif

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

//===============================================================================================================================================================================================================
//	DEFINE / INCLUDE
//===============================================================================================================================================================================================================
#define fp float

//===============================================================================================================================================================================================================
//	DEFINE / INCLUDE
//===============================================================================================================================================================================================================

fp* chop_flip_image(	char *image, 
								int height, 
								int width, 
								int cropped,
								int scaled,
								int converted) ;

fp* get_frame(	avi_t* cell_file, 
						int frame_num, 
						int cropped, 
						int scaled,
						int converted) ;

#ifdef __cplusplus
}
#endif

//======================================================================================================================================================
//	STRUCTURES, GLOBAL STRUCTURE VARIABLES
//======================================================================================================================================================

//===============================================================================================================================================================================================================
//===============================================================================================================================================================================================================
//	DEFINE / INCLUDE
//===============================================================================================================================================================================================================
//===============================================================================================================================================================================================================

// #define fp float

/* #define NUMBER_THREADS 512 */
#ifdef RD_WG_SIZE_0_0
        #define NUMBER_THREADS RD_WG_SIZE_0_0
#elif defined(RD_WG_SIZE_0)
        #define NUMBER_THREADS RD_WG_SIZE_0
#elif defined(RD_WG_SIZE)
        #define NUMBER_THREADS RD_WG_SIZE
#else
        #define NUMBER_THREADS 256
#endif


#define ENDO_POINTS 20
#define EPI_POINTS 31
#define ALL_POINTS 51

//===============================================================================================================================================================================================================
//===============================================================================================================================================================================================================
//	PARAMS_COMMON_CHANGE STRUCT
//===============================================================================================================================================================================================================
//===============================================================================================================================================================================================================

typedef struct params_common_change{

	//======================================================================================================================================================
	//	FRAME
	//======================================================================================================================================================

	fp* d_frame;
	int frame_no;

} params_common_change;

//===============================================================================================================================================================================================================
//===============================================================================================================================================================================================================
//	PARAMS_COMMON STRUCTURE
//===============================================================================================================================================================================================================
//===============================================================================================================================================================================================================

typedef struct params_common{

	//======================================================================================================================================================
	//	HARDCODED INPUTS FROM MATLAB
	//======================================================================================================================================================

	//====================================================================================================
	//	CONSTANTS
	//====================================================================================================

	int sSize;
	int tSize;
	int maxMove;
	fp alpha;

	//====================================================================================================
	//	FRAME
	//====================================================================================================

	int no_frames;
	int frame_rows;
	int frame_cols;
	int frame_elem;
	int frame_mem;

	//====================================================================================================
	//	ENDO POINTS
	//====================================================================================================

	int endoPoints;
	int endo_mem;

	int* endoRow;
	int* endoCol;
	int* tEndoRowLoc;
	int* tEndoColLoc;

	int* d_endoRow;
	int* d_endoCol;
	int* d_tEndoRowLoc;
	int* d_tEndoColLoc;

	fp* d_endoT;

	//====================================================================================================
	//	EPI POINTS
	//====================================================================================================
	int epiPoints;
	int epi_mem;

	int* epiRow;
	int* epiCol;
	int* tEpiRowLoc;
	int* tEpiColLoc;

	int* d_epiRow;
	int* d_epiCol;
	int* d_tEpiRowLoc;
	int* d_tEpiColLoc;

	fp* d_epiT;

	//====================================================================================================
	//	ALL POINTS
	//====================================================================================================

	int allPoints;

	//======================================================================================================================================================
	//	RIGHT TEMPLATE 	FROM 	TEMPLATE ARRAY
	//======================================================================================================================================================

	int in_rows;
	int in_cols;
	int in_elem;
	int in_mem;

	//======================================================================================================================================================
	// 	AREA AROUND POINT		FROM	FRAME
	//======================================================================================================================================================

	int in2_rows;
	int in2_cols;
	int in2_elem;
	int in2_mem;

	//======================================================================================================================================================
	//	CONVOLUTION
	//======================================================================================================================================================

	int conv_rows;
	int conv_cols;
	int conv_elem;
	int conv_mem;
	int ioffset;
	int joffset;

	//======================================================================================================================================================
	//	CUMULATIVE SUM 1
	//======================================================================================================================================================

	//====================================================================================================
	//	PAD ARRAY, VERTICAL CUMULATIVE SUM
	//====================================================================================================

	int in2_pad_add_rows;
	int in2_pad_add_cols;
	int in2_pad_cumv_rows;
	int in2_pad_cumv_cols;
	int in2_pad_cumv_elem;
	int in2_pad_cumv_mem;

	//====================================================================================================
	//	SELECTION
	//====================================================================================================

	int in2_pad_cumv_sel_rows;
	int in2_pad_cumv_sel_cols;
	int in2_pad_cumv_sel_elem;
	int in2_pad_cumv_sel_mem;
	int in2_pad_cumv_sel_rowlow;
	int in2_pad_cumv_sel_rowhig;
	int in2_pad_cumv_sel_collow;
	int in2_pad_cumv_sel_colhig;

	//====================================================================================================
	//	SELECTION 2, SUBTRACTION, HORIZONTAL CUMULATIVE SUM
	//====================================================================================================

	int in2_pad_cumv_sel2_rowlow;
	int in2_pad_cumv_sel2_rowhig;
	int in2_pad_cumv_sel2_collow;
	int in2_pad_cumv_sel2_colhig;
	int in2_sub_cumh_rows;
	int in2_sub_cumh_cols;
	int in2_sub_cumh_elem;
	int in2_sub_cumh_mem;

	//====================================================================================================
	//	SELECTION
	//====================================================================================================

	int in2_sub_cumh_sel_rows;
	int in2_sub_cumh_sel_cols;
	int in2_sub_cumh_sel_elem;
	int in2_sub_cumh_sel_mem;
	int in2_sub_cumh_sel_rowlow;
	int in2_sub_cumh_sel_rowhig;
	int in2_sub_cumh_sel_collow;
	int in2_sub_cumh_sel_colhig;

	//====================================================================================================
	//	SELECTION 2, SUBTRACTION
	//====================================================================================================

	int in2_sub_cumh_sel2_rowlow;
	int in2_sub_cumh_sel2_rowhig;
	int in2_sub_cumh_sel2_collow;
	int in2_sub_cumh_sel2_colhig;
	int in2_sub2_rows;
	int in2_sub2_cols;
	int in2_sub2_elem;
	int in2_sub2_mem;

	//======================================================================================================================================================
	//	CUMULATIVE SUM 2
	//======================================================================================================================================================

	//====================================================================================================
	//	MULTIPLICATION
	//====================================================================================================

	int in2_sqr_rows;
	int in2_sqr_cols;
	int in2_sqr_elem;
	int in2_sqr_mem;

	//====================================================================================================
	//	SELECTION 2, SUBTRACTION
	//====================================================================================================

	int in2_sqr_sub2_rows;
	int in2_sqr_sub2_cols;
	int in2_sqr_sub2_elem;
	int in2_sqr_sub2_mem;

	//======================================================================================================================================================
	//	FINAL
	//======================================================================================================================================================

	int in_sqr_rows;
	int in_sqr_cols;
	int in_sqr_elem;
	int in_sqr_mem;

	//======================================================================================================================================================
	//	TEMPLATE MASK CREATE
	//======================================================================================================================================================

	int tMask_rows;
	int tMask_cols;
	int tMask_elem;
	int tMask_mem;

	//======================================================================================================================================================
	//	POINT MASK INITIALIZE
	//======================================================================================================================================================

	int mask_rows;
	int mask_cols;
	int mask_elem;
	int mask_mem;

	//======================================================================================================================================================
	//	MASK CONVOLUTION
	//======================================================================================================================================================

	int mask_conv_rows;
	int mask_conv_cols;
	int mask_conv_elem;
	int mask_conv_mem;
	int mask_conv_ioffset;
	int mask_conv_joffset;

} params_common;

//===============================================================================================================================================================================================================
//===============================================================================================================================================================================================================
//	PARAMS_UNIQUE STRUCTURE
//===============================================================================================================================================================================================================
//===============================================================================================================================================================================================================

typedef struct params_unique{

	//======================================================================================================================================================
	//	POINT NUMBER
	//======================================================================================================================================================

	int* d_Row;
	int* d_Col;
	int* d_tRowLoc;
	int* d_tColLoc;
	fp* d_T;

	//======================================================================================================================================================
	//	POINT NUMBER
	//======================================================================================================================================================

	int point_no;

	//======================================================================================================================================================
	// 	RIGHT TEMPLATE 	FROM 	TEMPLATE ARRAY
	//======================================================================================================================================================

	int in_pointer;

	//======================================================================================================================================================
	//	AREA AROUND POINT		FROM	FRAME
	//======================================================================================================================================================

	fp* d_in2;

	//======================================================================================================================================================
	//	CONVOLUTION
	//======================================================================================================================================================

	fp* d_conv;
	fp* d_in_mod;

	//======================================================================================================================================================
	//	CUMULATIVE SUM
	//======================================================================================================================================================

	//====================================================================================================
	//	PAD ARRAY, VERTICAL CUMULATIVE SUM
	//====================================================================================================

	fp* d_in2_pad_cumv;

	//====================================================================================================
	//	SELECTION
	//====================================================================================================

	fp* d_in2_pad_cumv_sel;

	//====================================================================================================
	//	SELECTION 2, SUBTRACTION, HORIZONTAL CUMULATIVE SUM
	//====================================================================================================

	fp* d_in2_sub_cumh;

	//====================================================================================================
	//	SELECTION
	//====================================================================================================

	fp* d_in2_sub_cumh_sel;

	//====================================================================================================
	//	SELECTION 2, SUBTRACTION
	//====================================================================================================

	fp* d_in2_sub2;

	//======================================================================================================================================================
	//	CUMULATIVE SUM 2
	//======================================================================================================================================================

	//====================================================================================================
	//	MULTIPLICATION
	//====================================================================================================

	fp* d_in2_sqr;

	//====================================================================================================
	//	SELECTION 2, SUBTRACTION
	//====================================================================================================

	fp* d_in2_sqr_sub2;

	//======================================================================================================================================================
	//	FINAL
	//======================================================================================================================================================

	fp* d_in_sqr;

	//======================================================================================================================================================
	//	TEMPLATE MASK
	//======================================================================================================================================================

	fp* d_tMask;

	//======================================================================================================================================================
	//	POINT MASK INITIALIZE
	//======================================================================================================================================================

	fp* d_mask;

	//======================================================================================================================================================
	//	MASK CONVOLUTION
	//======================================================================================================================================================

	fp* d_mask_conv;

} params_unique;

//===============================================================================================================================================================================================================
//===============================================================================================================================================================================================================
//	END OF STRUCTURE
//===============================================================================================================================================================================================================
//===============================================================================================================================================================================================================

params_common_change common_change;
__constant__ params_common_change d_common_change;

params_common common;
__constant__ params_common d_common;

params_unique unique[ALL_POINTS];								// cannot determine size dynamically so choose more than usually needed
__constant__ params_unique d_unique[ALL_POINTS];

//======================================================================================================================================================
// KERNEL CODE
//======================================================================================================================================================

//===============================================================================================================================================================================================================
//===============================================================================================================================================================================================================
//	KERNEL FUNCTION
//===============================================================================================================================================================================================================
//===============================================================================================================================================================================================================

__global__ void kernel(){

	//======================================================================================================================================================
	//	COMMON VARIABLES
	//======================================================================================================================================================

	fp* d_in;
	int rot_row;
	int rot_col;
	int in2_rowlow;
	int in2_collow;
	int ic;
	int jc;
	int jp1;
	int ja1, ja2;
	int ip1;
	int ia1, ia2;
	int ja, jb;
	int ia, ib;
	float s;
	int i;
	int j;
	int row;
	int col;
	int ori_row;
	int ori_col;
	int position;
	float sum;
	int pos_ori;
	float temp;
	float temp2;
	int location;
	int cent;
	int tMask_row; 
	int tMask_col;
	float largest_value_current = 0;
	float largest_value = 0;
	int largest_coordinate_current = 0;
	int largest_coordinate = 0;
	float fin_max_val = 0;
	int fin_max_coo = 0;
	int largest_row;
	int largest_col;
	int offset_row;
	int offset_col;
	__shared__ float in_partial_sum[51];															// WATCH THIS !!! HARDCODED VALUE
	__shared__ float in_sqr_partial_sum[51];															// WATCH THIS !!! HARDCODED VALUE
	__shared__ float in_final_sum;
	__shared__ float in_sqr_final_sum;
	float mean;
	float mean_sqr;
	float variance;
	float deviation;
	__shared__ float denomT;
	__shared__ float par_max_val[131];															// WATCH THIS !!! HARDCODED VALUE
	__shared__ int par_max_coo[131];															// WATCH THIS !!! HARDCODED VALUE
	int pointer;
	__shared__ float d_in_mod_temp[2601];
	int ori_pointer;
	int loc_pointer;

	//======================================================================================================================================================
	//	THREAD PARAMETERS
	//======================================================================================================================================================

	int bx = blockIdx.x;																// get current horizontal block index (0-n)
	int tx = threadIdx.x;																// get current horizontal thread index (0-n)
	int ei_new;

	//===============================================================================================================================================================================================================
	//===============================================================================================================================================================================================================
	//	GENERATE TEMPLATE
	//===============================================================================================================================================================================================================
	//===============================================================================================================================================================================================================

	// generate templates based on the first frame only
	if(d_common_change.frame_no == 0){

		//======================================================================================================================================================
		// GET POINTER TO TEMPLATE FOR THE POINT
		//======================================================================================================================================================

		// pointers to: current template for current point
		d_in = &d_unique[bx].d_T[d_unique[bx].in_pointer];

		//======================================================================================================================================================
		//	UPDATE ROW LOC AND COL LOC
		//======================================================================================================================================================

		// uptade temporary endo/epi row/col coordinates (in each block corresponding to point, narrow work to one thread)
		ei_new = tx;
		if(ei_new == 0){

			// update temporary row/col coordinates
			pointer = d_unique[bx].point_no*d_common.no_frames+d_common_change.frame_no;
			d_unique[bx].d_tRowLoc[pointer] = d_unique[bx].d_Row[d_unique[bx].point_no];
			d_unique[bx].d_tColLoc[pointer] = d_unique[bx].d_Col[d_unique[bx].point_no];

		}

		//======================================================================================================================================================
		//	CREATE TEMPLATES
		//======================================================================================================================================================

		// work
		ei_new = tx;
		while(ei_new < d_common.in_elem){

			// figure out row/col location in new matrix
			row = (ei_new+1) % d_common.in_rows - 1;												// (0-n) row
			col = (ei_new+1) / d_common.in_rows + 1 - 1;											// (0-n) column
			if((ei_new+1) % d_common.in_rows == 0){
				row = d_common.in_rows - 1;
				col = col-1;
			}

			// figure out row/col location in corresponding new template area in image and give to every thread (get top left corner and progress down and right)
			ori_row = d_unique[bx].d_Row[d_unique[bx].point_no] - 25 + row - 1;
			ori_col = d_unique[bx].d_Col[d_unique[bx].point_no] - 25 + col - 1;
			ori_pointer = ori_col*d_common.frame_rows+ori_row;

			// update template
			d_in[col*d_common.in_rows+row] = d_common_change.d_frame[ori_pointer];

			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}

	}

	//===============================================================================================================================================================================================================
	//===============================================================================================================================================================================================================
	//	PROCESS POINTS
	//===============================================================================================================================================================================================================
	//===============================================================================================================================================================================================================

	// process points in all frames except for the first one
	if(d_common_change.frame_no != 0){

		//======================================================================================================================================================
		//	SELECTION
		//======================================================================================================================================================

		in2_rowlow = d_unique[bx].d_Row[d_unique[bx].point_no] - d_common.sSize;													// (1 to n+1)
		in2_collow = d_unique[bx].d_Col[d_unique[bx].point_no] - d_common.sSize;

		// work
		ei_new = tx;
		while(ei_new < d_common.in2_elem){

			// figure out row/col location in new matrix
			row = (ei_new+1) % d_common.in2_rows - 1;												// (0-n) row
			col = (ei_new+1) / d_common.in2_rows + 1 - 1;											// (0-n) column
			if((ei_new+1) % d_common.in2_rows == 0){
				row = d_common.in2_rows - 1;
				col = col-1;
			}

			// figure out corresponding location in old matrix and copy values to new matrix
			ori_row = row + in2_rowlow - 1;
			ori_col = col + in2_collow - 1;
			d_unique[bx].d_in2[ei_new] = d_common_change.d_frame[ori_col*d_common.frame_rows+ori_row];

			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}

		//======================================================================================================================================================
		//	SYNCHRONIZE THREADS
		//======================================================================================================================================================

		__syncthreads();

		//======================================================================================================================================================
		//	CONVOLUTION
		//======================================================================================================================================================

		//====================================================================================================
		//	ROTATION
		//====================================================================================================

		// variables
		d_in = &d_unique[bx].d_T[d_unique[bx].in_pointer];

		// work
		ei_new = tx;
		while(ei_new < d_common.in_elem){

			// figure out row/col location in padded array
			row = (ei_new+1) % d_common.in_rows - 1;												// (0-n) row
			col = (ei_new+1) / d_common.in_rows + 1 - 1;											// (0-n) column
			if((ei_new+1) % d_common.in_rows == 0){
				row = d_common.in_rows - 1;
				col = col-1;
			}
		
			// execution
			rot_row = (d_common.in_rows-1) - row;
			rot_col = (d_common.in_rows-1) - col;
			d_in_mod_temp[ei_new] = d_in[rot_col*d_common.in_rows+rot_row];

			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}

		//====================================================================================================
		//	SYNCHRONIZE THREADS
		//====================================================================================================

		__syncthreads();

		//====================================================================================================
		//	ACTUAL CONVOLUTION
		//====================================================================================================

		// work
		ei_new = tx;
		while(ei_new < d_common.conv_elem){

			// figure out row/col location in array
			ic = (ei_new+1) % d_common.conv_rows;												// (1-n)
			jc = (ei_new+1) / d_common.conv_rows + 1;											// (1-n)
			if((ei_new+1) % d_common.conv_rows == 0){
				ic = d_common.conv_rows;
				jc = jc-1;
			}

			//
			j = jc + d_common.joffset;
			jp1 = j + 1;
			if(d_common.in2_cols < jp1){
				ja1 = jp1 - d_common.in2_cols;
			}
			else{
				ja1 = 1;
			}
			if(d_common.in_cols < j){
				ja2 = d_common.in_cols;
			}
			else{
				ja2 = j;
			}

			i = ic + d_common.ioffset;
			ip1 = i + 1;
			
			if(d_common.in2_rows < ip1){
				ia1 = ip1 - d_common.in2_rows;
			}
			else{
				ia1 = 1;
			}
			if(d_common.in_rows < i){
				ia2 = d_common.in_rows;
			}
			else{
				ia2 = i;
			}

			s = 0;

			for(ja=ja1; ja<=ja2; ja++){
				jb = jp1 - ja;
				for(ia=ia1; ia<=ia2; ia++){
					ib = ip1 - ia;
					s = s + d_in_mod_temp[d_common.in_rows*(ja-1)+ia-1] * d_unique[bx].d_in2[d_common.in2_rows*(jb-1)+ib-1];
				}
			}

			//d_unique[bx].d_conv[d_common.conv_rows*(jc-1)+ic-1] = s;
			d_unique[bx].d_conv[ei_new] = s;

			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}

		//======================================================================================================================================================
		//	SYNCHRONIZE THREADS
		//======================================================================================================================================================

		__syncthreads();

		//======================================================================================================================================================
		//	CUMULATIVE SUM
		//======================================================================================================================================================

		//====================================================================================================
		//	PAD ARRAY, VERTICAL CUMULATIVE SUM
		//====================================================================================================

		//==================================================
		//	PADD ARRAY
		//==================================================

		// work
		ei_new = tx;
		while(ei_new < d_common.in2_pad_cumv_elem){

			// figure out row/col location in padded array
			row = (ei_new+1) % d_common.in2_pad_cumv_rows - 1;												// (0-n) row
			col = (ei_new+1) / d_common.in2_pad_cumv_rows + 1 - 1;											// (0-n) column
			if((ei_new+1) % d_common.in2_pad_cumv_rows == 0){
				row = d_common.in2_pad_cumv_rows - 1;
				col = col-1;
			}

			// execution
			if(	row > (d_common.in2_pad_add_rows-1) &&														// do if has numbers in original array
				row < (d_common.in2_pad_add_rows+d_common.in2_rows) && 
				col > (d_common.in2_pad_add_cols-1) && 
				col < (d_common.in2_pad_add_cols+d_common.in2_cols)){
				ori_row = row - d_common.in2_pad_add_rows;
				ori_col = col - d_common.in2_pad_add_cols;
				d_unique[bx].d_in2_pad_cumv[ei_new] = d_unique[bx].d_in2[ori_col*d_common.in2_rows+ori_row];
			}
			else{																			// do if otherwise
				d_unique[bx].d_in2_pad_cumv[ei_new] = 0;
			}

			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}

		//==================================================
		//	SYNCHRONIZE THREADS
		//==================================================

		__syncthreads();

		//==================================================
		//	VERTICAL CUMULATIVE SUM
		//==================================================

		//work
		ei_new = tx;
		while(ei_new < d_common.in2_pad_cumv_cols){

			// figure out column position
			pos_ori = ei_new*d_common.in2_pad_cumv_rows;

			// variables
			sum = 0;
			
			// loop through all rows
			for(position = pos_ori; position < pos_ori+d_common.in2_pad_cumv_rows; position = position + 1){
				d_unique[bx].d_in2_pad_cumv[position] = d_unique[bx].d_in2_pad_cumv[position] + sum;
				sum = d_unique[bx].d_in2_pad_cumv[position];
			}

			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}

		//====================================================================================================
		//	SYNCHRONIZE THREADS
		//====================================================================================================

		__syncthreads();

		//====================================================================================================
		//	SELECTION
		//====================================================================================================

		// work
		ei_new = tx;
		while(ei_new < d_common.in2_pad_cumv_sel_elem){

			// figure out row/col location in new matrix
			row = (ei_new+1) % d_common.in2_pad_cumv_sel_rows - 1;												// (0-n) row
			col = (ei_new+1) / d_common.in2_pad_cumv_sel_rows + 1 - 1;											// (0-n) column
			if((ei_new+1) % d_common.in2_pad_cumv_sel_rows == 0){
				row = d_common.in2_pad_cumv_sel_rows - 1;
				col = col-1;
			}

			// figure out corresponding location in old matrix and copy values to new matrix
			ori_row = row + d_common.in2_pad_cumv_sel_rowlow - 1;
			ori_col = col + d_common.in2_pad_cumv_sel_collow - 1;
			d_unique[bx].d_in2_pad_cumv_sel[ei_new] = d_unique[bx].d_in2_pad_cumv[ori_col*d_common.in2_pad_cumv_rows+ori_row];

			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}

		//====================================================================================================
		//	SYNCHRONIZE THREADS
		//====================================================================================================

		__syncthreads();

		//====================================================================================================
		//	SELECTION 2, SUBTRACTION, HORIZONTAL CUMULATIVE SUM
		//====================================================================================================

		//==================================================
		//	SELECTION 2
		//==================================================

		// work
		ei_new = tx;
		while(ei_new < d_common.in2_sub_cumh_elem){

			// figure out row/col location in new matrix
			row = (ei_new+1) % d_common.in2_sub_cumh_rows - 1;												// (0-n) row
			col = (ei_new+1) / d_common.in2_sub_cumh_rows + 1 - 1;											// (0-n) column
			if((ei_new+1) % d_common.in2_sub_cumh_rows == 0){
				row = d_common.in2_sub_cumh_rows - 1;
				col = col-1;
			}

			// figure out corresponding location in old matrix and copy values to new matrix
			ori_row = row + d_common.in2_pad_cumv_sel2_rowlow - 1;
			ori_col = col + d_common.in2_pad_cumv_sel2_collow - 1;
			d_unique[bx].d_in2_sub_cumh[ei_new] = d_unique[bx].d_in2_pad_cumv[ori_col*d_common.in2_pad_cumv_rows+ori_row];

			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}

		//==================================================
		//	SYNCHRONIZE THREADS
		//==================================================

		__syncthreads();

		//==================================================
		//	SUBTRACTION
		//==================================================
		
		// work
		ei_new = tx;
		while(ei_new < d_common.in2_sub_cumh_elem){

			// subtract
			d_unique[bx].d_in2_sub_cumh[ei_new] = d_unique[bx].d_in2_pad_cumv_sel[ei_new] - d_unique[bx].d_in2_sub_cumh[ei_new];

			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}

		//==================================================
		//	SYNCHRONIZE THREADS
		//==================================================

		__syncthreads();

		//==================================================
		//	HORIZONTAL CUMULATIVE SUM
		//==================================================

		// work
		ei_new = tx;
		while(ei_new < d_common.in2_sub_cumh_rows){

			// figure out row position
			pos_ori = ei_new;

			// variables
			sum = 0;

			// loop through all rows
			for(position = pos_ori; position < pos_ori+d_common.in2_sub_cumh_elem; position = position + d_common.in2_sub_cumh_rows){
				d_unique[bx].d_in2_sub_cumh[position] = d_unique[bx].d_in2_sub_cumh[position] + sum;
				sum = d_unique[bx].d_in2_sub_cumh[position];
			}

			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}

		//====================================================================================================
		//	SYNCHRONIZE THREADS
		//====================================================================================================

		__syncthreads();

		//====================================================================================================
		//	SELECTION
		//====================================================================================================

		// work
		ei_new = tx;
		while(ei_new < d_common.in2_sub_cumh_sel_elem){

			// figure out row/col location in new matrix
			row = (ei_new+1) % d_common.in2_sub_cumh_sel_rows - 1;												// (0-n) row
			col = (ei_new+1) / d_common.in2_sub_cumh_sel_rows + 1 - 1;											// (0-n) column
			if((ei_new+1) % d_common.in2_sub_cumh_sel_rows == 0){
				row = d_common.in2_sub_cumh_sel_rows - 1;
				col = col - 1;
			}

			// figure out corresponding location in old matrix and copy values to new matrix
			ori_row = row + d_common.in2_sub_cumh_sel_rowlow - 1;
			ori_col = col + d_common.in2_sub_cumh_sel_collow - 1;
			d_unique[bx].d_in2_sub_cumh_sel[ei_new] = d_unique[bx].d_in2_sub_cumh[ori_col*d_common.in2_sub_cumh_rows+ori_row];

			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}

		//====================================================================================================
		//	SYNCHRONIZE THREADS
		//====================================================================================================

		__syncthreads();

		//====================================================================================================
		//	SELECTION 2, SUBTRACTION
		//====================================================================================================

		//==================================================
		//	SELECTION 2
		//==================================================

		// work
		ei_new = tx;
		while(ei_new < d_common.in2_sub2_elem){

			// figure out row/col location in new matrix
			row = (ei_new+1) % d_common.in2_sub2_rows - 1;												// (0-n) row
			col = (ei_new+1) / d_common.in2_sub2_rows + 1 - 1;											// (0-n) column
			if((ei_new+1) % d_common.in2_sub2_rows == 0){
				row = d_common.in2_sub2_rows - 1;
				col = col-1;
			}

			// figure out corresponding location in old matrix and copy values to new matrix
			ori_row = row + d_common.in2_sub_cumh_sel2_rowlow - 1;
			ori_col = col + d_common.in2_sub_cumh_sel2_collow - 1;
			d_unique[bx].d_in2_sub2[ei_new] = d_unique[bx].d_in2_sub_cumh[ori_col*d_common.in2_sub_cumh_rows+ori_row];

			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}

		//==================================================
		//	SYNCHRONIZE THREADS
		//==================================================

		__syncthreads();

		//==================================================
		//	SUBTRACTION
		//==================================================

		// work
		ei_new = tx;
		while(ei_new < d_common.in2_sub2_elem){

			// subtract
			d_unique[bx].d_in2_sub2[ei_new] = d_unique[bx].d_in2_sub_cumh_sel[ei_new] - d_unique[bx].d_in2_sub2[ei_new];

			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}

		//======================================================================================================================================================
		//	SYNCHRONIZE THREADS
		//======================================================================================================================================================

		__syncthreads();

		//======================================================================================================================================================
		//	CUMULATIVE SUM 2
		//======================================================================================================================================================

		//====================================================================================================
		//	MULTIPLICATION
		//====================================================================================================

		// work
		ei_new = tx;
		while(ei_new < d_common.in2_sqr_elem){

			temp = d_unique[bx].d_in2[ei_new];
			d_unique[bx].d_in2_sqr[ei_new] = temp * temp;

			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}

		//====================================================================================================
		//	SYNCHRONIZE THREADS
		//====================================================================================================

		__syncthreads();

		//====================================================================================================
		//	PAD ARRAY, VERTICAL CUMULATIVE SUM
		//====================================================================================================

		//==================================================
		//	PAD ARRAY
		//==================================================

		// work
		ei_new = tx;
		while(ei_new < d_common.in2_pad_cumv_elem){

			// figure out row/col location in padded array
			row = (ei_new+1) % d_common.in2_pad_cumv_rows - 1;												// (0-n) row
			col = (ei_new+1) / d_common.in2_pad_cumv_rows + 1 - 1;											// (0-n) column
			if((ei_new+1) % d_common.in2_pad_cumv_rows == 0){
				row = d_common.in2_pad_cumv_rows - 1;
				col = col-1;
			}

			// execution
			if(	row > (d_common.in2_pad_add_rows-1) &&													// do if has numbers in original array
				row < (d_common.in2_pad_add_rows+d_common.in2_sqr_rows) && 
				col > (d_common.in2_pad_add_cols-1) && 
				col < (d_common.in2_pad_add_cols+d_common.in2_sqr_cols)){
				ori_row = row - d_common.in2_pad_add_rows;
				ori_col = col - d_common.in2_pad_add_cols;
				d_unique[bx].d_in2_pad_cumv[ei_new] = d_unique[bx].d_in2_sqr[ori_col*d_common.in2_sqr_rows+ori_row];
			}
			else{																							// do if otherwise
				d_unique[bx].d_in2_pad_cumv[ei_new] = 0;
			}

			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}

		//==================================================
		//	SYNCHRONIZE THREADS
		//==================================================

		__syncthreads();

		//==================================================
		//	VERTICAL CUMULATIVE SUM
		//==================================================

		//work
		ei_new = tx;
		while(ei_new < d_common.in2_pad_cumv_cols){

			// figure out column position
			pos_ori = ei_new*d_common.in2_pad_cumv_rows;

			// variables
			sum = 0;
			
			// loop through all rows
			for(position = pos_ori; position < pos_ori+d_common.in2_pad_cumv_rows; position = position + 1){
				d_unique[bx].d_in2_pad_cumv[position] = d_unique[bx].d_in2_pad_cumv[position] + sum;
				sum = d_unique[bx].d_in2_pad_cumv[position];
			}

			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}

		//====================================================================================================
		//	SYNCHRONIZE THREADS
		//====================================================================================================

		__syncthreads();

		//====================================================================================================
		//	SELECTION
		//====================================================================================================

		// work
		ei_new = tx;
		while(ei_new < d_common.in2_pad_cumv_sel_elem){

			// figure out row/col location in new matrix
			row = (ei_new+1) % d_common.in2_pad_cumv_sel_rows - 1;												// (0-n) row
			col = (ei_new+1) / d_common.in2_pad_cumv_sel_rows + 1 - 1;											// (0-n) column
			if((ei_new+1) % d_common.in2_pad_cumv_sel_rows == 0){
				row = d_common.in2_pad_cumv_sel_rows - 1;
				col = col-1;
			}

			// figure out corresponding location in old matrix and copy values to new matrix
			ori_row = row + d_common.in2_pad_cumv_sel_rowlow - 1;
			ori_col = col + d_common.in2_pad_cumv_sel_collow - 1;
			d_unique[bx].d_in2_pad_cumv_sel[ei_new] = d_unique[bx].d_in2_pad_cumv[ori_col*d_common.in2_pad_cumv_rows+ori_row];

			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}

		//====================================================================================================
		//	SYNCHRONIZE THREADS
		//====================================================================================================

		__syncthreads();

		//====================================================================================================
		//	SELECTION 2, SUBTRACTION, HORIZONTAL CUMULATIVE SUM
		//====================================================================================================

		//==================================================
		//	SELECTION 2
		//==================================================

		// work
		ei_new = tx;
		while(ei_new < d_common.in2_sub_cumh_elem){

			// figure out row/col location in new matrix
			row = (ei_new+1) % d_common.in2_sub_cumh_rows - 1;												// (0-n) row
			col = (ei_new+1) / d_common.in2_sub_cumh_rows + 1 - 1;											// (0-n) column
			if((ei_new+1) % d_common.in2_sub_cumh_rows == 0){
				row = d_common.in2_sub_cumh_rows - 1;
				col = col-1;
			}

			// figure out corresponding location in old matrix and copy values to new matrix
			ori_row = row + d_common.in2_pad_cumv_sel2_rowlow - 1;
			ori_col = col + d_common.in2_pad_cumv_sel2_collow - 1;
			d_unique[bx].d_in2_sub_cumh[ei_new] = d_unique[bx].d_in2_pad_cumv[ori_col*d_common.in2_pad_cumv_rows+ori_row];

			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}

		//==================================================
		//	SYNCHRONIZE THREADS
		//==================================================

		__syncthreads();

		//==================================================
		//	SUBTRACTION
		//==================================================

		// work
		ei_new = tx;
		while(ei_new < d_common.in2_sub_cumh_elem){

			// subtract
			d_unique[bx].d_in2_sub_cumh[ei_new] = d_unique[bx].d_in2_pad_cumv_sel[ei_new] - d_unique[bx].d_in2_sub_cumh[ei_new];

			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}

		//==================================================
		//	HORIZONTAL CUMULATIVE SUM
		//==================================================

		// work
		ei_new = tx;
		while(ei_new < d_common.in2_sub_cumh_rows){

			// figure out row position
			pos_ori = ei_new;

			// variables
			sum = 0;

			// loop through all rows
			for(position = pos_ori; position < pos_ori+d_common.in2_sub_cumh_elem; position = position + d_common.in2_sub_cumh_rows){
				d_unique[bx].d_in2_sub_cumh[position] = d_unique[bx].d_in2_sub_cumh[position] + sum;
				sum = d_unique[bx].d_in2_sub_cumh[position];
			}

			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}

		//====================================================================================================
		//	SYNCHRONIZE THREADS
		//====================================================================================================

		__syncthreads();

		//====================================================================================================
		//	SELECTION
		//====================================================================================================

		// work
		ei_new = tx;
		while(ei_new < d_common.in2_sub_cumh_sel_elem){

			// figure out row/col location in new matrix
			row = (ei_new+1) % d_common.in2_sub_cumh_sel_rows - 1;												// (0-n) row
			col = (ei_new+1) / d_common.in2_sub_cumh_sel_rows + 1 - 1;											// (0-n) column
			if((ei_new+1) % d_common.in2_sub_cumh_sel_rows == 0){
				row = d_common.in2_sub_cumh_sel_rows - 1;
				col = col - 1;
			}

			// figure out corresponding location in old matrix and copy values to new matrix
			ori_row = row + d_common.in2_sub_cumh_sel_rowlow - 1;
			ori_col = col + d_common.in2_sub_cumh_sel_collow - 1;
			d_unique[bx].d_in2_sub_cumh_sel[ei_new] = d_unique[bx].d_in2_sub_cumh[ori_col*d_common.in2_sub_cumh_rows+ori_row];

			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}

		//====================================================================================================
		//	SYNCHRONIZE THREADS
		//====================================================================================================

		__syncthreads();

		//====================================================================================================
		//	SELECTION 2, SUBTRACTION
		//====================================================================================================

		//==================================================
		//	SELECTION 2
		//==================================================

		// work
		ei_new = tx;
		while(ei_new < d_common.in2_sub2_elem){

			// figure out row/col location in new matrix
			row = (ei_new+1) % d_common.in2_sub2_rows - 1;												// (0-n) row
			col = (ei_new+1) / d_common.in2_sub2_rows + 1 - 1;											// (0-n) column
			if((ei_new+1) % d_common.in2_sub2_rows == 0){
				row = d_common.in2_sub2_rows - 1;
				col = col-1;
			}

			// figure out corresponding location in old matrix and copy values to new matrix
			ori_row = row + d_common.in2_sub_cumh_sel2_rowlow - 1;
			ori_col = col + d_common.in2_sub_cumh_sel2_collow - 1;
			d_unique[bx].d_in2_sqr_sub2[ei_new] = d_unique[bx].d_in2_sub_cumh[ori_col*d_common.in2_sub_cumh_rows+ori_row];

			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}

		//==================================================
		//	SYNCHRONIZE THREADS
		//==================================================

		__syncthreads();

		//==================================================
		//	SUBTRACTION
		//==================================================

		// work
		ei_new = tx;
		while(ei_new < d_common.in2_sub2_elem){

			// subtract
			d_unique[bx].d_in2_sqr_sub2[ei_new] = d_unique[bx].d_in2_sub_cumh_sel[ei_new] - d_unique[bx].d_in2_sqr_sub2[ei_new];

			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}

		//======================================================================================================================================================
		//	SYNCHRONIZE THREADS
		//======================================================================================================================================================

		__syncthreads();

		//======================================================================================================================================================
		//	FINAL
		//======================================================================================================================================================

		//====================================================================================================
		//	DENOMINATOR A		SAVE RESULT IN CUMULATIVE SUM A2
		//====================================================================================================

		// work
		ei_new = tx;
		while(ei_new < d_common.in2_sub2_elem){

			temp = d_unique[bx].d_in2_sub2[ei_new];
			temp2 = d_unique[bx].d_in2_sqr_sub2[ei_new] - (temp * temp / d_common.in_elem);
			if(temp2 < 0){
				temp2 = 0;
			}
			d_unique[bx].d_in2_sqr_sub2[ei_new] = sqrt(temp2);
			

			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}

		//====================================================================================================
		//	SYNCHRONIZE THREADS
		//====================================================================================================

		__syncthreads();

		//====================================================================================================
		//	MULTIPLICATION
		//====================================================================================================

		// work
		ei_new = tx;
		while(ei_new < d_common.in_sqr_elem){

			temp = d_in[ei_new];
			d_unique[bx].d_in_sqr[ei_new] = temp * temp;

			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}

		//====================================================================================================
		//	SYNCHRONIZE THREADS
		//====================================================================================================

		__syncthreads();

		//====================================================================================================
		//	IN SUM
		//====================================================================================================

		// work
		ei_new = tx;
		while(ei_new < d_common.in_cols){

			sum = 0;
			for(i = 0; i < d_common.in_rows; i++){

				sum = sum + d_in[ei_new*d_common.in_rows+i];

			}
			in_partial_sum[ei_new] = sum;

			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}

		//====================================================================================================
		//	SYNCHRONIZE THREADS
		//====================================================================================================

		__syncthreads();

		//====================================================================================================
		//	IN_SQR SUM
		//====================================================================================================

		ei_new = tx;
		while(ei_new < d_common.in_sqr_rows){
				
			sum = 0;
			for(i = 0; i < d_common.in_sqr_cols; i++){

				sum = sum + d_unique[bx].d_in_sqr[ei_new+d_common.in_sqr_rows*i];

			}
			in_sqr_partial_sum[ei_new] = sum;

			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}

		//====================================================================================================
		//	SYNCHRONIZE THREADS
		//====================================================================================================

		__syncthreads();

		//====================================================================================================
		//	FINAL SUMMATION
		//====================================================================================================

		if(tx == 0){

			in_final_sum = 0;
			for(i = 0; i<d_common.in_cols; i++){
				in_final_sum = in_final_sum + in_partial_sum[i];
			}

		}else if(tx == 1){

			in_sqr_final_sum = 0;
			for(i = 0; i<d_common.in_sqr_cols; i++){
				in_sqr_final_sum = in_sqr_final_sum + in_sqr_partial_sum[i];
			}

		}

		//====================================================================================================
		//	SYNCHRONIZE THREADS
		//====================================================================================================

		__syncthreads();

		//====================================================================================================
		//	DENOMINATOR T
		//====================================================================================================

		if(tx == 0){

			mean = in_final_sum / d_common.in_elem;													// gets mean (average) value of element in ROI
			mean_sqr = mean * mean;
			variance  = (in_sqr_final_sum / d_common.in_elem) - mean_sqr;							// gets variance of ROI
			deviation = sqrt(variance);																// gets standard deviation of ROI

			denomT = sqrt(float(d_common.in_elem-1))*deviation;

		}

		//====================================================================================================
		//	SYNCHRONIZE THREADS
		//====================================================================================================

		__syncthreads();

		//====================================================================================================
		//	DENOMINATOR		SAVE RESULT IN CUMULATIVE SUM A2
		//====================================================================================================

		// work
		ei_new = tx;
		while(ei_new < d_common.in2_sub2_elem){

			d_unique[bx].d_in2_sqr_sub2[ei_new] = d_unique[bx].d_in2_sqr_sub2[ei_new] * denomT;

			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}

		//====================================================================================================
		//	SYNCHRONIZE THREADS
		//====================================================================================================

		__syncthreads();

		//====================================================================================================
		//	NUMERATOR	SAVE RESULT IN CONVOLUTION
		//====================================================================================================

		// work
		ei_new = tx;
		while(ei_new < d_common.conv_elem){

			d_unique[bx].d_conv[ei_new] = d_unique[bx].d_conv[ei_new] - d_unique[bx].d_in2_sub2[ei_new] * in_final_sum / d_common.in_elem;

			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}

		//====================================================================================================
		//	SYNCHRONIZE THREADS
		//====================================================================================================

		__syncthreads();

		//====================================================================================================
		//	CORRELATION	SAVE RESULT IN CUMULATIVE SUM A2
		//====================================================================================================

		// work
		ei_new = tx;
		while(ei_new < d_common.in2_sub2_elem){

			d_unique[bx].d_in2_sqr_sub2[ei_new] = d_unique[bx].d_conv[ei_new] / d_unique[bx].d_in2_sqr_sub2[ei_new];

			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}

		//======================================================================================================================================================
		//	SYNCHRONIZE THREADS
		//======================================================================================================================================================

		__syncthreads();

		//======================================================================================================================================================
		//	TEMPLATE MASK CREATE
		//======================================================================================================================================================

		cent = d_common.sSize + d_common.tSize + 1;
		if(d_common_change.frame_no == 0){
			tMask_row = cent + d_unique[bx].d_Row[d_unique[bx].point_no] - d_unique[bx].d_Row[d_unique[bx].point_no] - 1;
			tMask_col = cent + d_unique[bx].d_Col[d_unique[bx].point_no] - d_unique[bx].d_Col[d_unique[bx].point_no] - 1;
		}
		else{
			pointer = d_common_change.frame_no-1+d_unique[bx].point_no*d_common.no_frames;
			tMask_row = cent + d_unique[bx].d_tRowLoc[pointer] - d_unique[bx].d_Row[d_unique[bx].point_no] - 1;
			tMask_col = cent + d_unique[bx].d_tColLoc[pointer] - d_unique[bx].d_Col[d_unique[bx].point_no] - 1;
		}


		//work
		ei_new = tx;
		while(ei_new < d_common.tMask_elem){

			location = tMask_col*d_common.tMask_rows + tMask_row;

			if(ei_new==location){
				d_unique[bx].d_tMask[ei_new] = 1;
			}
			else{
				d_unique[bx].d_tMask[ei_new] = 0;
			}

			//go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}

		//======================================================================================================================================================
		//	SYNCHRONIZE THREADS
		//======================================================================================================================================================

		__syncthreads();

		//======================================================================================================================================================
		//	MASK CONVOLUTION
		//======================================================================================================================================================

		// work
		ei_new = tx;
		while(ei_new < d_common.mask_conv_elem){

			// figure out row/col location in array
			ic = (ei_new+1) % d_common.mask_conv_rows;												// (1-n)
			jc = (ei_new+1) / d_common.mask_conv_rows + 1;											// (1-n)
			if((ei_new+1) % d_common.mask_conv_rows == 0){
				ic = d_common.mask_conv_rows;
				jc = jc-1;
			}

			//
			j = jc + d_common.mask_conv_joffset;
			jp1 = j + 1;
			if(d_common.mask_cols < jp1){
				ja1 = jp1 - d_common.mask_cols;
			}
			else{
				ja1 = 1;
			}
			if(d_common.tMask_cols < j){
				ja2 = d_common.tMask_cols;
			}
			else{
				ja2 = j;
			}

			i = ic + d_common.mask_conv_ioffset;
			ip1 = i + 1;
			
			if(d_common.mask_rows < ip1){
				ia1 = ip1 - d_common.mask_rows;
			}
			else{
				ia1 = 1;
			}
			if(d_common.tMask_rows < i){
				ia2 = d_common.tMask_rows;
			}
			else{
				ia2 = i;
			}

			s = 0;

			for(ja=ja1; ja<=ja2; ja++){
				jb = jp1 - ja;
				for(ia=ia1; ia<=ia2; ia++){
					ib = ip1 - ia;
					s = s + d_unique[bx].d_tMask[d_common.tMask_rows*(ja-1)+ia-1] * 1;
				}
			}

			// //d_unique[bx].d_mask_conv[d_common.mask_conv_rows*(jc-1)+ic-1] = s;
			d_unique[bx].d_mask_conv[ei_new] = d_unique[bx].d_in2_sqr_sub2[ei_new] * s;

			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}

		//======================================================================================================================================================
		//	SYNCHRONIZE THREADS
		//======================================================================================================================================================

		__syncthreads();

		//======================================================================================================================================================
		//	MAXIMUM VALUE
		//======================================================================================================================================================

		//====================================================================================================
		//	INITIAL SEARCH
		//====================================================================================================

		ei_new = tx;
		while(ei_new < d_common.mask_conv_rows){

			for(i=0; i<d_common.mask_conv_cols; i++){
				largest_coordinate_current = ei_new*d_common.mask_conv_rows+i;
				largest_value_current = abs(d_unique[bx].d_mask_conv[largest_coordinate_current]);
				if(largest_value_current > largest_value){
					largest_coordinate = largest_coordinate_current;
					largest_value = largest_value_current;
				}
			}
			par_max_coo[ei_new] = largest_coordinate;
			par_max_val[ei_new] = largest_value;

			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}

		//====================================================================================================
		//	SYNCHRONIZE THREADS
		//====================================================================================================

		__syncthreads();

		//====================================================================================================
		//	FINAL SEARCH
		//====================================================================================================

		if(tx == 0){

			for(i = 0; i < d_common.mask_conv_rows; i++){
				if(par_max_val[i] > fin_max_val){
					fin_max_val = par_max_val[i];
					fin_max_coo = par_max_coo[i];
				}
			}

			// convert coordinate to row/col form
			largest_row = (fin_max_coo+1) % d_common.mask_conv_rows - 1;											// (0-n) row
			largest_col = (fin_max_coo+1) / d_common.mask_conv_rows;												// (0-n) column
			if((fin_max_coo+1) % d_common.mask_conv_rows == 0){
				largest_row = d_common.mask_conv_rows - 1;
				largest_col = largest_col - 1;
			}

			// calculate offset
			largest_row = largest_row + 1;																	// compensate to match MATLAB format (1-n)
			largest_col = largest_col + 1;																	// compensate to match MATLAB format (1-n)
			offset_row = largest_row - d_common.in_rows - (d_common.sSize - d_common.tSize);
			offset_col = largest_col - d_common.in_cols - (d_common.sSize - d_common.tSize);
			pointer = d_common_change.frame_no+d_unique[bx].point_no*d_common.no_frames;
			d_unique[bx].d_tRowLoc[pointer] = d_unique[bx].d_Row[d_unique[bx].point_no] + offset_row;
			d_unique[bx].d_tColLoc[pointer] = d_unique[bx].d_Col[d_unique[bx].point_no] + offset_col;

		}

		//======================================================================================================================================================
		//	SYNCHRONIZE THREADS
		//======================================================================================================================================================

		__syncthreads();

	}
	
	//===============================================================================================================================================================================================================
	//===============================================================================================================================================================================================================
	//	COORDINATE AND TEMPLATE UPDATE
	//===============================================================================================================================================================================================================
	//===============================================================================================================================================================================================================

	// time19 = clock();

	// if the last frame in the bath, update template
	if(d_common_change.frame_no != 0 && (d_common_change.frame_no)%10 == 0){

		// update coordinate
		loc_pointer = d_unique[bx].point_no*d_common.no_frames+d_common_change.frame_no;
		d_unique[bx].d_Row[d_unique[bx].point_no] = d_unique[bx].d_tRowLoc[loc_pointer];
		d_unique[bx].d_Col[d_unique[bx].point_no] = d_unique[bx].d_tColLoc[loc_pointer];

		// work
		ei_new = tx;
		while(ei_new < d_common.in_elem){

			// figure out row/col location in new matrix
			row = (ei_new+1) % d_common.in_rows - 1;												// (0-n) row
			col = (ei_new+1) / d_common.in_rows + 1 - 1;											// (0-n) column
			if((ei_new+1) % d_common.in_rows == 0){
				row = d_common.in_rows - 1;
				col = col-1;
			}

			// figure out row/col location in corresponding new template area in image and give to every thread (get top left corner and progress down and right)
			ori_row = d_unique[bx].d_Row[d_unique[bx].point_no] - 25 + row - 1;
			ori_col = d_unique[bx].d_Col[d_unique[bx].point_no] - 25 + col - 1;
			ori_pointer = ori_col*d_common.frame_rows+ori_row;

			// update template
			d_in[ei_new] = d_common.alpha*d_in[ei_new] + (1.00-d_common.alpha)*d_common_change.d_frame[ori_pointer];

			// go for second round
			ei_new = ei_new + NUMBER_THREADS;

		}

	}

}

	//===============================================================================================================================================================================================================
	//===============================================================================================================================================================================================================
	//	END OF FUNCTION
	//===============================================================================================================================================================================================================
	//===============================================================================================================================================================================================================






//	WRITE DATA FUNCTION
//===============================================================================================================================================================================================================200

void write_data(	char* filename,
			int frameNo,
			int frames_processed,
			int endoPoints,
			int* input_a,
			int* input_b,
			int epiPoints,
			int* input_2a,
			int* input_2b){

	//================================================================================80
	//	VARIABLES
	//================================================================================80

	FILE* fid;
	int i,j;
	char c;

	//================================================================================80
	//	OPEN FILE FOR READING
	//================================================================================80

	fid = fopen(filename, "w+");
	if( fid == NULL ){
		printf( "The file was not opened for writing\n" );
		return;
	}


	//================================================================================80
	//	WRITE VALUES TO THE FILE
	//================================================================================80
      fprintf(fid, "Total AVI Frames: %d\n", frameNo);	
      fprintf(fid, "Frames Processed: %d\n", frames_processed);	
      fprintf(fid, "endoPoints: %d\n", endoPoints);
      fprintf(fid, "epiPoints: %d", epiPoints);
	for(j=0; j<frames_processed;j++)
	  {
	    fprintf(fid, "\n---Frame %d---",j);
	    fprintf(fid, "\n--endo--\n",j);
	    for(i=0; i<endoPoints; i++){
	      fprintf(fid, "%d\t", input_a[j+i*frameNo]);
	    }
	    fprintf(fid, "\n");
	    for(i=0; i<endoPoints; i++){
	      // if(input_b[j*size+i] > 2000) input_b[j*size+i]=0;
	      fprintf(fid, "%d\t", input_b[j+i*frameNo]);
	    }
	    fprintf(fid, "\n--epi--\n",j);
	    for(i=0; i<epiPoints; i++){
	      //if(input_2a[j*size_2+i] > 2000) input_2a[j*size_2+i]=0;
	      fprintf(fid, "%d\t", input_2a[j+i*frameNo]);
	    }
	    fprintf(fid, "\n");
	    for(i=0; i<epiPoints; i++){
	      //if(input_2b[j*size_2+i] > 2000) input_2b[j*size_2+i]=0;
	      fprintf(fid, "%d\t", input_2b[j+i*frameNo]);
	    }
	  }
	// 	================================================================================80
	//		CLOSE FILE
		  //	================================================================================80

	fclose(fid);

}


//===============================================================================================================================================================================================================
//===============================================================================================================================================================================================================
//	MAIN FUNCTION
//===============================================================================================================================================================================================================
//===============================================================================================================================================================================================================
int main(int argc, char *argv []){

  printf("WG size of kernel = %d \n", NUMBER_THREADS);
	//======================================================================================================================================================
	//	VARIABLES
	//======================================================================================================================================================

	// CUDA kernel execution parameters
	dim3 threads;
	dim3 blocks;

	// counter
	int i;
	int frames_processed;

	// frames
	char* video_file_name;
	avi_t* frames;
	fp* frame;

	//======================================================================================================================================================
	// 	FRAME
	//======================================================================================================================================================

	if(argc!=3){
		printf("ERROR: usage: heartwall <inputfile> <num of frames>\n");
		exit(1);
	}
	
	// open movie file
 	video_file_name = argv[1];
	frames = (avi_t*)AVI_open_input_file(video_file_name, 1);														// added casting
	if (frames == NULL)  {
		   AVI_print_error((char *) "Error with AVI_open_input_file");
		   return -1;
	}

	// common
	common.no_frames = AVI_video_frames(frames);
	common.frame_rows = AVI_video_height(frames);
	common.frame_cols = AVI_video_width(frames);
	common.frame_elem = common.frame_rows * common.frame_cols;
	common.frame_mem = sizeof(fp) * common.frame_elem;

	// pointers
	cudaMalloc((void **)&common_change.d_frame, common.frame_mem);

	//======================================================================================================================================================
	// 	CHECK INPUT ARGUMENTS
	//======================================================================================================================================================
	
	frames_processed = atoi(argv[2]);
		if(frames_processed<0 || frames_processed>common.no_frames){
			printf("ERROR: %d is an incorrect number of frames specified, select in the range of 0-%d\n", frames_processed, common.no_frames);
			return 0;
	}
	

	//======================================================================================================================================================
	//	HARDCODED INPUTS FROM MATLAB
	//======================================================================================================================================================

	//====================================================================================================
	//	CONSTANTS
	//====================================================================================================

	common.sSize = 40;
	common.tSize = 25;
	common.maxMove = 10;
	common.alpha = 0.87;

	//====================================================================================================
	//	ENDO POINTS
	//====================================================================================================

	common.endoPoints = ENDO_POINTS;
	common.endo_mem = sizeof(int) * common.endoPoints;

	common.endoRow = (int *)malloc(common.endo_mem);
	common.endoRow[ 0] = 369;
	common.endoRow[ 1] = 400;
	common.endoRow[ 2] = 429;
	common.endoRow[ 3] = 452;
	common.endoRow[ 4] = 476;
	common.endoRow[ 5] = 486;
	common.endoRow[ 6] = 479;
	common.endoRow[ 7] = 458;
	common.endoRow[ 8] = 433;
	common.endoRow[ 9] = 404;
	common.endoRow[10] = 374;
	common.endoRow[11] = 346;
	common.endoRow[12] = 318;
	common.endoRow[13] = 294;
	common.endoRow[14] = 277;
	common.endoRow[15] = 269;
	common.endoRow[16] = 275;
	common.endoRow[17] = 287;
	common.endoRow[18] = 311;
	common.endoRow[19] = 339;
	cudaMalloc((void **)&common.d_endoRow, common.endo_mem);
	cudaMemcpy(common.d_endoRow, common.endoRow, common.endo_mem, cudaMemcpyHostToDevice);

	common.endoCol = (int *)malloc(common.endo_mem);
	common.endoCol[ 0] = 408;
	common.endoCol[ 1] = 406;
	common.endoCol[ 2] = 397;
	common.endoCol[ 3] = 383;
	common.endoCol[ 4] = 354;
	common.endoCol[ 5] = 322;
	common.endoCol[ 6] = 294;
	common.endoCol[ 7] = 270;
	common.endoCol[ 8] = 250;
	common.endoCol[ 9] = 237;
	common.endoCol[10] = 235;
	common.endoCol[11] = 241;
	common.endoCol[12] = 254;
	common.endoCol[13] = 273;
	common.endoCol[14] = 300;
	common.endoCol[15] = 328;
	common.endoCol[16] = 356;
	common.endoCol[17] = 383;
	common.endoCol[18] = 401;
	common.endoCol[19] = 411;
	cudaMalloc((void **)&common.d_endoCol, common.endo_mem);
	cudaMemcpy(common.d_endoCol, common.endoCol, common.endo_mem, cudaMemcpyHostToDevice);

	common.tEndoRowLoc = (int *)malloc(common.endo_mem * common.no_frames);
	cudaMalloc((void **)&common.d_tEndoRowLoc, common.endo_mem * common.no_frames);

	common.tEndoColLoc = (int *)malloc(common.endo_mem * common.no_frames);
	cudaMalloc((void **)&common.d_tEndoColLoc, common.endo_mem * common.no_frames);

	//====================================================================================================
	//	EPI POINTS
	//====================================================================================================

	common.epiPoints = EPI_POINTS;
	common.epi_mem = sizeof(int) * common.epiPoints;

	common.epiRow = (int *)malloc(common.epi_mem);
	common.epiRow[ 0] = 390;
	common.epiRow[ 1] = 419;
	common.epiRow[ 2] = 448;
	common.epiRow[ 3] = 474;
	common.epiRow[ 4] = 501;
	common.epiRow[ 5] = 519;
	common.epiRow[ 6] = 535;
	common.epiRow[ 7] = 542;
	common.epiRow[ 8] = 543;
	common.epiRow[ 9] = 538;
	common.epiRow[10] = 528;
	common.epiRow[11] = 511;
	common.epiRow[12] = 491;
	common.epiRow[13] = 466;
	common.epiRow[14] = 438;
	common.epiRow[15] = 406;
	common.epiRow[16] = 376;
	common.epiRow[17] = 347;
	common.epiRow[18] = 318;
	common.epiRow[19] = 291;
	common.epiRow[20] = 275;
	common.epiRow[21] = 259;
	common.epiRow[22] = 256;
	common.epiRow[23] = 252;
	common.epiRow[24] = 252;
	common.epiRow[25] = 257;
	common.epiRow[26] = 266;
	common.epiRow[27] = 283;
	common.epiRow[28] = 305;
	common.epiRow[29] = 331;
	common.epiRow[30] = 360;
	cudaMalloc((void **)&common.d_epiRow, common.epi_mem);
	cudaMemcpy(common.d_epiRow, common.epiRow, common.epi_mem, cudaMemcpyHostToDevice);

	common.epiCol = (int *)malloc(common.epi_mem);
	common.epiCol[ 0] = 457;
	common.epiCol[ 1] = 454;
	common.epiCol[ 2] = 446;
	common.epiCol[ 3] = 431;
	common.epiCol[ 4] = 411;
	common.epiCol[ 5] = 388;
	common.epiCol[ 6] = 361;
	common.epiCol[ 7] = 331;
	common.epiCol[ 8] = 301;
	common.epiCol[ 9] = 273;
	common.epiCol[10] = 243;
	common.epiCol[11] = 218;
	common.epiCol[12] = 196;
	common.epiCol[13] = 178;
	common.epiCol[14] = 166;
	common.epiCol[15] = 157;
	common.epiCol[16] = 155;
	common.epiCol[17] = 165;
	common.epiCol[18] = 177;
	common.epiCol[19] = 197;
	common.epiCol[20] = 218;
	common.epiCol[21] = 248;
	common.epiCol[22] = 276;
	common.epiCol[23] = 304;
	common.epiCol[24] = 333;
	common.epiCol[25] = 361;
	common.epiCol[26] = 391;
	common.epiCol[27] = 415;
	common.epiCol[28] = 434;
	common.epiCol[29] = 448;
	common.epiCol[30] = 455;
	cudaMalloc((void **)&common.d_epiCol, common.epi_mem);
	cudaMemcpy(common.d_epiCol, common.epiCol, common.epi_mem, cudaMemcpyHostToDevice);

	common.tEpiRowLoc = (int *)malloc(common.epi_mem * common.no_frames);
	cudaMalloc((void **)&common.d_tEpiRowLoc, common.epi_mem * common.no_frames);

	common.tEpiColLoc = (int *)malloc(common.epi_mem * common.no_frames);
	cudaMalloc((void **)&common.d_tEpiColLoc, common.epi_mem * common.no_frames);

	//====================================================================================================
	//	ALL POINTS
	//====================================================================================================

	common.allPoints = ALL_POINTS;

	//======================================================================================================================================================
	// 	TEMPLATE SIZES
	//======================================================================================================================================================

	// common
	common.in_rows = common.tSize + 1 + common.tSize;
	common.in_cols = common.in_rows;
	common.in_elem = common.in_rows * common.in_cols;
	common.in_mem = sizeof(fp) * common.in_elem;

	//======================================================================================================================================================
	// 	CREATE ARRAY OF TEMPLATES FOR ALL POINTS
	//======================================================================================================================================================

	// common
	cudaMalloc((void **)&common.d_endoT, common.in_mem * common.endoPoints);
	cudaMalloc((void **)&common.d_epiT, common.in_mem * common.epiPoints);

	//======================================================================================================================================================
	//	SPECIFIC TO ENDO OR EPI TO BE SET HERE
	//======================================================================================================================================================

	for(i=0; i<common.endoPoints; i++){
		unique[i].point_no = i;
		unique[i].d_Row = common.d_endoRow;
		unique[i].d_Col = common.d_endoCol;
		unique[i].d_tRowLoc = common.d_tEndoRowLoc;
		unique[i].d_tColLoc = common.d_tEndoColLoc;
		unique[i].d_T = common.d_endoT;
	}
	for(i=common.endoPoints; i<common.allPoints; i++){
		unique[i].point_no = i-common.endoPoints;
		unique[i].d_Row = common.d_epiRow;
		unique[i].d_Col = common.d_epiCol;
		unique[i].d_tRowLoc = common.d_tEpiRowLoc;
		unique[i].d_tColLoc = common.d_tEpiColLoc;
		unique[i].d_T = common.d_epiT;
	}

	//======================================================================================================================================================
	// 	RIGHT TEMPLATE 	FROM 	TEMPLATE ARRAY
	//======================================================================================================================================================

	// pointers
	for(i=0; i<common.allPoints; i++){
		unique[i].in_pointer = unique[i].point_no * common.in_elem;
	}

	//======================================================================================================================================================
	// 	AREA AROUND POINT		FROM	FRAME
	//======================================================================================================================================================

	// common
	common.in2_rows = 2 * common.sSize + 1;
	common.in2_cols = 2 * common.sSize + 1;
	common.in2_elem = common.in2_rows * common.in2_cols;
	common.in2_mem = sizeof(float) * common.in2_elem;

	// pointers
	for(i=0; i<common.allPoints; i++){
		cudaMalloc((void **)&unique[i].d_in2, common.in2_mem);
	}

	//======================================================================================================================================================
	// 	CONVOLUTION
	//======================================================================================================================================================

	// common
	common.conv_rows = common.in_rows + common.in2_rows - 1;												// number of rows in I
	common.conv_cols = common.in_cols + common.in2_cols - 1;												// number of columns in I
	common.conv_elem = common.conv_rows * common.conv_cols;												// number of elements
	common.conv_mem = sizeof(float) * common.conv_elem;
	common.ioffset = 0;
	common.joffset = 0;

	// pointers
	for(i=0; i<common.allPoints; i++){
		cudaMalloc((void **)&unique[i].d_conv, common.conv_mem);
	}

	//======================================================================================================================================================
	// 	CUMULATIVE SUM
	//======================================================================================================================================================

	//====================================================================================================
	// 	PADDING OF ARRAY, VERTICAL CUMULATIVE SUM
	//====================================================================================================

	// common
	common.in2_pad_add_rows = common.in_rows;
	common.in2_pad_add_cols = common.in_cols;

	common.in2_pad_cumv_rows = common.in2_rows + 2*common.in2_pad_add_rows;
	common.in2_pad_cumv_cols = common.in2_cols + 2*common.in2_pad_add_cols;
	common.in2_pad_cumv_elem = common.in2_pad_cumv_rows * common.in2_pad_cumv_cols;
	common.in2_pad_cumv_mem = sizeof(float) * common.in2_pad_cumv_elem;

	// pointers
	for(i=0; i<common.allPoints; i++){
		cudaMalloc((void **)&unique[i].d_in2_pad_cumv, common.in2_pad_cumv_mem);
	}

	//====================================================================================================
	// 	SELECTION
	//====================================================================================================

	// common
	common.in2_pad_cumv_sel_rowlow = 1 + common.in_rows;													// (1 to n+1)
	common.in2_pad_cumv_sel_rowhig = common.in2_pad_cumv_rows - 1;
	common.in2_pad_cumv_sel_collow = 1;
	common.in2_pad_cumv_sel_colhig = common.in2_pad_cumv_cols;
	common.in2_pad_cumv_sel_rows = common.in2_pad_cumv_sel_rowhig - common.in2_pad_cumv_sel_rowlow + 1;
	common.in2_pad_cumv_sel_cols = common.in2_pad_cumv_sel_colhig - common.in2_pad_cumv_sel_collow + 1;
	common.in2_pad_cumv_sel_elem = common.in2_pad_cumv_sel_rows * common.in2_pad_cumv_sel_cols;
	common.in2_pad_cumv_sel_mem = sizeof(float) * common.in2_pad_cumv_sel_elem;

	// pointers
	for(i=0; i<common.allPoints; i++){
		cudaMalloc((void **)&unique[i].d_in2_pad_cumv_sel, common.in2_pad_cumv_sel_mem);
	}

	//====================================================================================================
	// 	SELECTION	2, SUBTRACTION, HORIZONTAL CUMULATIVE SUM
	//====================================================================================================

	// common
	common.in2_pad_cumv_sel2_rowlow = 1;
	common.in2_pad_cumv_sel2_rowhig = common.in2_pad_cumv_rows - common.in_rows - 1;
	common.in2_pad_cumv_sel2_collow = 1;
	common.in2_pad_cumv_sel2_colhig = common.in2_pad_cumv_cols;
	common.in2_sub_cumh_rows = common.in2_pad_cumv_sel2_rowhig - common.in2_pad_cumv_sel2_rowlow + 1;
	common.in2_sub_cumh_cols = common.in2_pad_cumv_sel2_colhig - common.in2_pad_cumv_sel2_collow + 1;
	common.in2_sub_cumh_elem = common.in2_sub_cumh_rows * common.in2_sub_cumh_cols;
	common.in2_sub_cumh_mem = sizeof(float) * common.in2_sub_cumh_elem;

	// pointers
	for(i=0; i<common.allPoints; i++){
		cudaMalloc((void **)&unique[i].d_in2_sub_cumh, common.in2_sub_cumh_mem);
	}

	//====================================================================================================
	// 	SELECTION
	//====================================================================================================

	// common
	common.in2_sub_cumh_sel_rowlow = 1;
	common.in2_sub_cumh_sel_rowhig = common.in2_sub_cumh_rows;
	common.in2_sub_cumh_sel_collow = 1 + common.in_cols;
	common.in2_sub_cumh_sel_colhig = common.in2_sub_cumh_cols - 1;
	common.in2_sub_cumh_sel_rows = common.in2_sub_cumh_sel_rowhig - common.in2_sub_cumh_sel_rowlow + 1;
	common.in2_sub_cumh_sel_cols = common.in2_sub_cumh_sel_colhig - common.in2_sub_cumh_sel_collow + 1;
	common.in2_sub_cumh_sel_elem = common.in2_sub_cumh_sel_rows * common.in2_sub_cumh_sel_cols;
	common.in2_sub_cumh_sel_mem = sizeof(float) * common.in2_sub_cumh_sel_elem;

	// pointers
	for(i=0; i<common.allPoints; i++){
		cudaMalloc((void **)&unique[i].d_in2_sub_cumh_sel, common.in2_sub_cumh_sel_mem);
	}

	//====================================================================================================
	//	SELECTION 2, SUBTRACTION
	//====================================================================================================

	// common
	common.in2_sub_cumh_sel2_rowlow = 1;
	common.in2_sub_cumh_sel2_rowhig = common.in2_sub_cumh_rows;
	common.in2_sub_cumh_sel2_collow = 1;
	common.in2_sub_cumh_sel2_colhig = common.in2_sub_cumh_cols - common.in_cols - 1;
	common.in2_sub2_rows = common.in2_sub_cumh_sel2_rowhig - common.in2_sub_cumh_sel2_rowlow + 1;
	common.in2_sub2_cols = common.in2_sub_cumh_sel2_colhig - common.in2_sub_cumh_sel2_collow + 1;
	common.in2_sub2_elem = common.in2_sub2_rows * common.in2_sub2_cols;
	common.in2_sub2_mem = sizeof(float) * common.in2_sub2_elem;

	// pointers
	for(i=0; i<common.allPoints; i++){
		cudaMalloc((void **)&unique[i].d_in2_sub2, common.in2_sub2_mem);
	}

	//======================================================================================================================================================
	//	CUMULATIVE SUM 2
	//======================================================================================================================================================

	//====================================================================================================
	//	MULTIPLICATION
	//====================================================================================================

	// common
	common.in2_sqr_rows = common.in2_rows;
	common.in2_sqr_cols = common.in2_cols;
	common.in2_sqr_elem = common.in2_elem;
	common.in2_sqr_mem = common.in2_mem;

	// pointers
	for(i=0; i<common.allPoints; i++){
		cudaMalloc((void **)&unique[i].d_in2_sqr, common.in2_sqr_mem);
	}

	//====================================================================================================
	//	SELECTION 2, SUBTRACTION
	//====================================================================================================

	// common
	common.in2_sqr_sub2_rows = common.in2_sub2_rows;
	common.in2_sqr_sub2_cols = common.in2_sub2_cols;
	common.in2_sqr_sub2_elem = common.in2_sub2_elem;
	common.in2_sqr_sub2_mem = common.in2_sub2_mem;

	// pointers
	for(i=0; i<common.allPoints; i++){
		cudaMalloc((void **)&unique[i].d_in2_sqr_sub2, common.in2_sqr_sub2_mem);
	}

	//======================================================================================================================================================
	//	FINAL
	//======================================================================================================================================================

	// common
	common.in_sqr_rows = common.in_rows;
	common.in_sqr_cols = common.in_cols;
	common.in_sqr_elem = common.in_elem;
	common.in_sqr_mem = common.in_mem;

	// pointers
	for(i=0; i<common.allPoints; i++){
		cudaMalloc((void **)&unique[i].d_in_sqr, common.in_sqr_mem);
	}

	//======================================================================================================================================================
	//	TEMPLATE MASK CREATE
	//======================================================================================================================================================

	// common
	common.tMask_rows = common.in_rows + (common.sSize+1+common.sSize) - 1;
	common.tMask_cols = common.tMask_rows;
	common.tMask_elem = common.tMask_rows * common.tMask_cols;
	common.tMask_mem = sizeof(float) * common.tMask_elem;

	// pointers
	for(i=0; i<common.allPoints; i++){
		cudaMalloc((void **)&unique[i].d_tMask, common.tMask_mem);
	}

	//======================================================================================================================================================
	//	POINT MASK INITIALIZE
	//======================================================================================================================================================

	// common
	common.mask_rows = common.maxMove;
	common.mask_cols = common.mask_rows;
	common.mask_elem = common.mask_rows * common.mask_cols;
	common.mask_mem = sizeof(float) * common.mask_elem;

	//======================================================================================================================================================
	//	MASK CONVOLUTION
	//======================================================================================================================================================

	// common
	common.mask_conv_rows = common.tMask_rows;												// number of rows in I
	common.mask_conv_cols = common.tMask_cols;												// number of columns in I
	common.mask_conv_elem = common.mask_conv_rows * common.mask_conv_cols;												// number of elements
	common.mask_conv_mem = sizeof(float) * common.mask_conv_elem;
	common.mask_conv_ioffset = (common.mask_rows-1)/2;
	if((common.mask_rows-1) % 2 > 0.5){
		common.mask_conv_ioffset = common.mask_conv_ioffset + 1;
	}
	common.mask_conv_joffset = (common.mask_cols-1)/2;
	if((common.mask_cols-1) % 2 > 0.5){
		common.mask_conv_joffset = common.mask_conv_joffset + 1;
	}

	// pointers
	for(i=0; i<common.allPoints; i++){
		cudaMalloc((void **)&unique[i].d_mask_conv, common.mask_conv_mem);
	}

	//======================================================================================================================================================
	//	KERNEL
	//======================================================================================================================================================

	//====================================================================================================
	//	THREAD BLOCK
	//====================================================================================================

	// All kernels operations within kernel use same max size of threads. Size of block size is set to the size appropriate for max size operation (on padded matrix). Other use subsets of that.
	threads.x = NUMBER_THREADS;											// define the number of threads in the block
	threads.y = 1;
	blocks.x = common.allPoints;							// define the number of blocks in the grid
	blocks.y = 1;

	//====================================================================================================
	//	COPY ARGUMENTS
	//====================================================================================================

	cudaMemcpyToSymbol(d_common, &common, sizeof(params_common));
	cudaMemcpyToSymbol(d_unique, &unique, sizeof(params_unique)*ALL_POINTS);

	//====================================================================================================
	//	PRINT FRAME PROGRESS START
	//====================================================================================================

	printf("frame progress: ");
	fflush(NULL);

	//====================================================================================================
	//	LAUNCH
	//====================================================================================================

	for(common_change.frame_no=0; common_change.frame_no<frames_processed; common_change.frame_no++){

		// Extract a cropped version of the first frame from the video file
		frame = get_frame(	frames,						// pointer to video file
										common_change.frame_no,				// number of frame that needs to be returned
										0,								// cropped?
										0,								// scaled?
										1);							// converted

		// copy frame to GPU memory
		cudaMemcpy(common_change.d_frame, frame, common.frame_mem, cudaMemcpyHostToDevice);
		cudaMemcpyToSymbol(d_common_change, &common_change, sizeof(params_common_change));

		// launch GPU kernel
		kernel<<<blocks, threads>>>();

		// free frame after each loop iteration, since AVI library allocates memory for every frame fetched
		free(frame);

		// print frame progress
		printf("%d ", common_change.frame_no);
		fflush(NULL);

	}

	//====================================================================================================
	//	PRINT FRAME PROGRESS END
	//====================================================================================================

	printf("\n");
	fflush(NULL);

	//====================================================================================================
	//	OUTPUT
	//====================================================================================================

	cudaMemcpy(common.tEndoRowLoc, common.d_tEndoRowLoc, common.endo_mem * common.no_frames, cudaMemcpyDeviceToHost);
	cudaMemcpy(common.tEndoColLoc, common.d_tEndoColLoc, common.endo_mem * common.no_frames, cudaMemcpyDeviceToHost);

	cudaMemcpy(common.tEpiRowLoc, common.d_tEpiRowLoc, common.epi_mem * common.no_frames, cudaMemcpyDeviceToHost);
	cudaMemcpy(common.tEpiColLoc, common.d_tEpiColLoc, common.epi_mem * common.no_frames, cudaMemcpyDeviceToHost);



#ifdef OUTPUT

	//==================================================50
	//	DUMP DATA TO FILE
	//==================================================50
	write_data(	"result.txt",
			common.no_frames,
			frames_processed,		
				common.endoPoints,
				common.tEndoRowLoc,
				common.tEndoColLoc,
				common.epiPoints,
				common.tEpiRowLoc,
				common.tEpiColLoc);

	//==================================================50
	//	End
	//==================================================50

#endif



	//======================================================================================================================================================
	//	DEALLOCATION
	//======================================================================================================================================================

	//====================================================================================================
	//	COMMON
	//====================================================================================================

	// frame
	cudaFree(common_change.d_frame);

	// endo points
	free(common.endoRow);
	free(common.endoCol);
	free(common.tEndoRowLoc);
	free(common.tEndoColLoc);

	cudaFree(common.d_endoRow);
	cudaFree(common.d_endoCol);
	cudaFree(common.d_tEndoRowLoc);
	cudaFree(common.d_tEndoColLoc);

	cudaFree(common.d_endoT);

	// epi points
	free(common.epiRow);
	free(common.epiCol);
	free(common.tEpiRowLoc);
	free(common.tEpiColLoc);

	cudaFree(common.d_epiRow);
	cudaFree(common.d_epiCol);
	cudaFree(common.d_tEpiRowLoc);
	cudaFree(common.d_tEpiColLoc);

	cudaFree(common.d_epiT);

	//====================================================================================================
	//	POINTERS
	//====================================================================================================

	for(i=0; i<common.allPoints; i++){
		cudaFree(unique[i].d_in2);

		cudaFree(unique[i].d_conv);
		cudaFree(unique[i].d_in2_pad_cumv);
		cudaFree(unique[i].d_in2_pad_cumv_sel);
		cudaFree(unique[i].d_in2_sub_cumh);
		cudaFree(unique[i].d_in2_sub_cumh_sel);
		cudaFree(unique[i].d_in2_sub2);
		cudaFree(unique[i].d_in2_sqr);
		cudaFree(unique[i].d_in2_sqr_sub2);
		cudaFree(unique[i].d_in_sqr);

		cudaFree(unique[i].d_tMask);
		cudaFree(unique[i].d_mask_conv);
	}

}

//===============================================================================================================================================================================================================
//===============================================================================================================================================================================================================
//	MAIN FUNCTION
//===============================================================================================================================================================================================================
//===============================================================================================================================================================================================================

#ifdef __cplusplus
extern "C" {
#endif

/*
 *  avilib.c
 *
 *  Copyright (C) Thomas �streich - June 2001
 *  multiple audio track support Copyright (C) 2002 Thomas �streich 
 *
 *  Original code:
 *  Copyright (C) 1999 Rainer Johanni <Rainer@Johanni.de> 
 *
 *  This file is part of transcode, a linux video stream processing tool
 *      
 *  transcode is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2, or (at your option)
 *  any later version.
 *   
 *  transcode is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *   
 *  You should have received a copy of the GNU General Public License
 *  along with GNU Make; see the file COPYING.  If not, write to
 *  the Free Software Foundation, 675 Mass Ave, Cambridge, MA 02139, USA. 
 *
 */

//#include "avilib.h"
//#include <time.h>

#define INFO_LIST

/* The following variable indicates the kind of error */

long AVI_errno;

#define MAX_INFO_STRLEN 64
static char id_str[MAX_INFO_STRLEN];

#define FRAME_RATE_SCALE 1000000

#ifndef PACKAGE
	#define PACKAGE "my"
	#define VERSION "0.00"
#endif

#ifndef O_BINARY
	/* win32 wants a binary flag to open(); this sets it to null
	   on platforms that don't have it. */
	#define O_BINARY 0
#endif

/*******************************************************************
 *                                                                 *
 *    Utilities for writing an AVI File                            *
 *                                                                 *
 *******************************************************************/

static size_t avi_read(int fd, char *buf, size_t len)
{
   size_t n = 0;
   size_t r = 0;

   while (r < len) {
      n = read (fd, buf + r, len - r);

      if (n <= 0)
	  return r;
      r += n;
   }

   return r;
}

static size_t avi_write (int fd, char *buf, size_t len)
{
   size_t n = 0;
   size_t r = 0;

   while (r < len) {
      n = write (fd, buf + r, len - r);
      if (n < 0)
         return n;
      
      r += n;
   }
   return r;
}

/* HEADERBYTES: The number of bytes to reserve for the header */

#define HEADERBYTES 2048

/* AVI_MAX_LEN: The maximum length of an AVI file, we stay a bit below
    the 2GB limit (Remember: 2*10^9 is smaller than 2 GB) */

#define AVI_MAX_LEN (UINT_MAX-(1<<20)*16-HEADERBYTES)

#define PAD_EVEN(x) ( ((x)+1) & ~1 )


/* Copy n into dst as a 4 byte, little endian number.
   Should also work on big endian machines */

static void long2str(unsigned char *dst, int n)
{
   dst[0] = (n    )&0xff;
   dst[1] = (n>> 8)&0xff;
   dst[2] = (n>>16)&0xff;
   dst[3] = (n>>24)&0xff;
}

/* Convert a string of 4 or 2 bytes to a number,
   also working on big endian machines */

static unsigned long str2ulong(unsigned char *str)
{
   return ( str[0] | (str[1]<<8) | (str[2]<<16) | (str[3]<<24) );
}
static unsigned long str2ushort(unsigned char *str)
{
   return ( str[0] | (str[1]<<8) );
}

/* Calculate audio sample size from number of bits and number of channels.
   This may have to be adjusted for eg. 12 bits and stereo */

static int avi_sampsize(avi_t *AVI, int j)
{
   int s;
   s = ((AVI->track[j].a_bits+7)/8)*AVI->track[j].a_chans;
   //   if(s==0) s=1; /* avoid possible zero divisions */
   if(s<4) s=4; /* avoid possible zero divisions */ 
   return s;
}

/* Add a chunk (=tag and data) to the AVI file,
   returns -1 on write error, 0 on success */

static int avi_add_chunk(avi_t *AVI, unsigned char *tag, unsigned char *data, int length)
{
   unsigned char c[8];

   /* Copy tag and length int c, so that we need only 1 write system call
      for these two values */

   memcpy(c,tag,4);
   long2str(c+4,length);

   /* Output tag, length and data, restore previous position
      if the write fails */

   length = PAD_EVEN(length);

   if( avi_write(AVI->fdes,(char *)c,8) != 8 ||
       avi_write(AVI->fdes,(char *)data,length) != length )
   {
      lseek(AVI->fdes,AVI->pos,SEEK_SET);
      AVI_errno = AVI_ERR_WRITE;
      return -1;
   }

   /* Update file position */

   AVI->pos += 8 + length;

   //fprintf(stderr, "pos=%lu %s\n", AVI->pos, tag);

   return 0;
}

static int avi_add_index_entry(avi_t *AVI, unsigned char *tag, long flags, unsigned long pos, unsigned long len)
{
   void *ptr;

   if(AVI->n_idx>=AVI->max_idx) {
     ptr = realloc((void *)AVI->idx,(AVI->max_idx+4096)*16);
     
     if(ptr == 0) {
       AVI_errno = AVI_ERR_NO_MEM;
       return -1;
     }
     AVI->max_idx += 4096;
     AVI->idx = (unsigned char((*)[16]) ) ptr;
   }
   
   /* Add index entry */

   //   fprintf(stderr, "INDEX %s %ld %lu %lu\n", tag, flags, pos, len);

   memcpy(AVI->idx[AVI->n_idx],tag,4);
   long2str(AVI->idx[AVI->n_idx]+ 4,flags);
   long2str(AVI->idx[AVI->n_idx]+ 8, pos);
   long2str(AVI->idx[AVI->n_idx]+12, len);
   
   /* Update counter */

   AVI->n_idx++;

   if(len>AVI->max_len) AVI->max_len=len;

   return 0;
}

/*
   AVI_open_output_file: Open an AVI File and write a bunch
                         of zero bytes as space for the header.

   returns a pointer to avi_t on success, a zero pointer on error
*/

avi_t* AVI_open_output_file(char * filename)
{
   avi_t *AVI;
   int i;

   int mask = 0;
   
   unsigned char AVI_header[HEADERBYTES];

   /* Allocate the avi_t struct and zero it */

   AVI = (avi_t *) malloc(sizeof(avi_t));
   if(AVI==0)
   {
      AVI_errno = AVI_ERR_NO_MEM;
      return 0;
   }
   memset((void *)AVI,0,sizeof(avi_t));

   /* Since Linux needs a long time when deleting big files,
      we do not truncate the file when we open it.
      Instead it is truncated when the AVI file is closed */

  /* mask = umask (0);
   umask (mask);*/

   AVI->fdes = open(filename, O_RDWR|O_CREAT|O_BINARY, 0644 &~ mask);
   if (AVI->fdes < 0)
   {
      AVI_errno = AVI_ERR_OPEN;
      free(AVI);
      return 0;
   }

   /* Write out HEADERBYTES bytes, the header will go here
      when we are finished with writing */

   for (i=0;i<HEADERBYTES;i++) AVI_header[i] = 0;
   i = avi_write(AVI->fdes,(char *)AVI_header,HEADERBYTES);
   if (i != HEADERBYTES)
   {
      close(AVI->fdes);
      AVI_errno = AVI_ERR_WRITE;
      free(AVI);
      return 0;
   }

   AVI->pos  = HEADERBYTES;
   AVI->mode = AVI_MODE_WRITE; /* open for writing */

   //init
   AVI->anum = 0;
   AVI->aptr = 0;

   return AVI;
}

void AVI_set_video(avi_t *AVI, int width, int height, double fps, char *compressor)
{
   /* may only be called if file is open for writing */

   if(AVI->mode==AVI_MODE_READ) return;

   AVI->width  = width;
   AVI->height = height;
   AVI->fps    = fps;
   
   if(strncmp(compressor, "RGB", 3)==0) {
     memset(AVI->compressor, 0, 4);
   } else {
     memcpy(AVI->compressor,compressor,4);
   }     
   
   AVI->compressor[4] = 0;

   avi_update_header(AVI);
}

void AVI_set_audio(avi_t *AVI, int channels, long rate, int bits, int format, long mp3rate)
{
   /* may only be called if file is open for writing */

   if(AVI->mode==AVI_MODE_READ) return;

   //inc audio tracks
   AVI->aptr=AVI->anum;
   ++AVI->anum;

   if(AVI->anum > AVI_MAX_TRACKS) {
     fprintf(stderr, "error - only %d audio tracks supported\n", AVI_MAX_TRACKS);
     exit(1);
   }

   AVI->track[AVI->aptr].a_chans = channels;
   AVI->track[AVI->aptr].a_rate  = rate;
   AVI->track[AVI->aptr].a_bits  = bits;
   AVI->track[AVI->aptr].a_fmt   = format;
   AVI->track[AVI->aptr].mp3rate = mp3rate;

   avi_update_header(AVI);
}

#define OUT4CC(s) \
   if(nhb<=HEADERBYTES-4) memcpy(AVI_header+nhb,s,4); nhb += 4

#define OUTLONG(n) \
   if(nhb<=HEADERBYTES-4) long2str(AVI_header+nhb,n); nhb += 4

#define OUTSHRT(n) \
   if(nhb<=HEADERBYTES-2) { \
      AVI_header[nhb  ] = (n   )&0xff; \
      AVI_header[nhb+1] = (n>>8)&0xff; \
   } \
   nhb += 2


//ThOe write preliminary AVI file header: 0 frames, max vid/aud size
int avi_update_header(avi_t *AVI)
{
   int njunk, sampsize, hasIndex, ms_per_frame, frate, flag;
   int movi_len, hdrl_start, strl_start, j;
   unsigned char AVI_header[HEADERBYTES];
   long nhb;

   //assume max size
   movi_len = AVI_MAX_LEN - HEADERBYTES + 4;

   //assume index will be written
   hasIndex=1;

   if(AVI->fps < 0.001) {
     frate=0;
     ms_per_frame=0;
   } else {
     frate = (int) (FRAME_RATE_SCALE*AVI->fps + 0.5);
     ms_per_frame=(int) (1000000/AVI->fps + 0.5);
   }

   /* Prepare the file header */

   nhb = 0;

   /* The RIFF header */

   OUT4CC ("RIFF");
   OUTLONG(movi_len);    // assume max size
   OUT4CC ("AVI ");

   /* Start the header list */

   OUT4CC ("LIST");
   OUTLONG(0);        /* Length of list in bytes, don't know yet */
   hdrl_start = nhb;  /* Store start position */
   OUT4CC ("hdrl");

   /* The main AVI header */

   /* The Flags in AVI File header */

#define AVIF_HASINDEX           0x00000010      /* Index at end of file */
#define AVIF_MUSTUSEINDEX       0x00000020
#define AVIF_ISINTERLEAVED      0x00000100
#define AVIF_TRUSTCKTYPE        0x00000800      /* Use CKType to find key frames */
#define AVIF_WASCAPTUREFILE     0x00010000
#define AVIF_COPYRIGHTED        0x00020000

   OUT4CC ("avih");
   OUTLONG(56);                 /* # of bytes to follow */
   OUTLONG(ms_per_frame);       /* Microseconds per frame */
   //ThOe ->0 
   //   OUTLONG(10000000);           /* MaxBytesPerSec, I hope this will never be used */
   OUTLONG(0);
   OUTLONG(0);                  /* PaddingGranularity (whatever that might be) */
                                /* Other sources call it 'reserved' */
   flag = AVIF_ISINTERLEAVED;
   if(hasIndex) flag |= AVIF_HASINDEX;
   if(hasIndex && AVI->must_use_index) flag |= AVIF_MUSTUSEINDEX;
   OUTLONG(flag);               /* Flags */
   OUTLONG(0);                  // no frames yet
   OUTLONG(0);                  /* InitialFrames */

   OUTLONG(AVI->anum+1);

   OUTLONG(0);                  /* SuggestedBufferSize */
   OUTLONG(AVI->width);         /* Width */
   OUTLONG(AVI->height);        /* Height */
                                /* MS calls the following 'reserved': */
   OUTLONG(0);                  /* TimeScale:  Unit used to measure time */
   OUTLONG(0);                  /* DataRate:   Data rate of playback     */
   OUTLONG(0);                  /* StartTime:  Starting time of AVI data */
   OUTLONG(0);                  /* DataLength: Size of AVI data chunk    */


   /* Start the video stream list ---------------------------------- */

   OUT4CC ("LIST");
   OUTLONG(0);        /* Length of list in bytes, don't know yet */
   strl_start = nhb;  /* Store start position */
   OUT4CC ("strl");

   /* The video stream header */

   OUT4CC ("strh");
   OUTLONG(56);                 /* # of bytes to follow */
   OUT4CC ("vids");             /* Type */
   OUT4CC (AVI->compressor);    /* Handler */
   OUTLONG(0);                  /* Flags */
   OUTLONG(0);                  /* Reserved, MS says: wPriority, wLanguage */
   OUTLONG(0);                  /* InitialFrames */
   OUTLONG(FRAME_RATE_SCALE);              /* Scale */
   OUTLONG(frate);              /* Rate: Rate/Scale == samples/second */
   OUTLONG(0);                  /* Start */
   OUTLONG(0);                  // no frames yet
   OUTLONG(0);                  /* SuggestedBufferSize */
   OUTLONG(-1);                 /* Quality */
   OUTLONG(0);                  /* SampleSize */
   OUTLONG(0);                  /* Frame */
   OUTLONG(0);                  /* Frame */
   //   OUTLONG(0);                  /* Frame */
   //OUTLONG(0);                  /* Frame */

   /* The video stream format */

   OUT4CC ("strf");
   OUTLONG(40);                 /* # of bytes to follow */
   OUTLONG(40);                 /* Size */
   OUTLONG(AVI->width);         /* Width */
   OUTLONG(AVI->height);        /* Height */
   OUTSHRT(1); OUTSHRT(24);     /* Planes, Count */
   OUT4CC (AVI->compressor);    /* Compression */
   // ThOe (*3)
   OUTLONG(AVI->width*AVI->height*3);  /* SizeImage (in bytes?) */
   OUTLONG(0);                  /* XPelsPerMeter */
   OUTLONG(0);                  /* YPelsPerMeter */
   OUTLONG(0);                  /* ClrUsed: Number of colors used */
   OUTLONG(0);                  /* ClrImportant: Number of colors important */

   /* Finish stream list, i.e. put number of bytes in the list to proper pos */

   long2str(AVI_header+strl_start-4,nhb-strl_start);

   
   /* Start the audio stream list ---------------------------------- */
   
   for(j=0; j<AVI->anum; ++j) {
       
       sampsize = avi_sampsize(AVI, j);
   
       OUT4CC ("LIST");
       OUTLONG(0);        /* Length of list in bytes, don't know yet */
       strl_start = nhb;  /* Store start position */
       OUT4CC ("strl");
       
       /* The audio stream header */
       
       OUT4CC ("strh");
       OUTLONG(56);            /* # of bytes to follow */
       OUT4CC ("auds");
       
       // -----------
       // ThOe
       OUTLONG(0);             /* Format (Optionally) */
       // -----------
       
       OUTLONG(0);             /* Flags */
       OUTLONG(0);             /* Reserved, MS says: wPriority, wLanguage */
       OUTLONG(0);             /* InitialFrames */
       
       // ThOe /4
       OUTLONG(sampsize/4);      /* Scale */
       OUTLONG(1000*AVI->track[j].mp3rate/8);
       OUTLONG(0);             /* Start */
       OUTLONG(4*AVI->track[j].audio_bytes/sampsize);   /* Length */
       OUTLONG(0);             /* SuggestedBufferSize */
       OUTLONG(-1);            /* Quality */
       
       // ThOe /4
       OUTLONG(sampsize/4);    /* SampleSize */
       
       OUTLONG(0);             /* Frame */
       OUTLONG(0);             /* Frame */
       //       OUTLONG(0);             /* Frame */
       //OUTLONG(0);             /* Frame */
       
       /* The audio stream format */
       
       OUT4CC ("strf");
       OUTLONG(16);                   /* # of bytes to follow */
       OUTSHRT(AVI->track[j].a_fmt);           /* Format */
       OUTSHRT(AVI->track[j].a_chans);         /* Number of channels */
       OUTLONG(AVI->track[j].a_rate);          /* SamplesPerSec */
       // ThOe
       OUTLONG(1000*AVI->track[j].mp3rate/8);
       //ThOe (/4)
       
       OUTSHRT(sampsize/4);           /* BlockAlign */
       
       
       OUTSHRT(AVI->track[j].a_bits);          /* BitsPerSample */
       
       /* Finish stream list, i.e. put number of bytes in the list to proper pos */
       
       long2str(AVI_header+strl_start-4,nhb-strl_start);
   }
   
   /* Finish header list */
   
   long2str(AVI_header+hdrl_start-4,nhb-hdrl_start);
   
   
   /* Calculate the needed amount of junk bytes, output junk */
   
   njunk = HEADERBYTES - nhb - 8 - 12;
   
   /* Safety first: if njunk <= 0, somebody has played with
      HEADERBYTES without knowing what (s)he did.
      This is a fatal error */
   
   if(njunk<=0)
     {
       fprintf(stderr,"AVI_close_output_file: # of header bytes too small\n");
       exit(1);
     }
   
   OUT4CC ("JUNK");
   OUTLONG(njunk);
   memset(AVI_header+nhb,0,njunk);
   
   //11/14/01 added id string 

   if(njunk > strlen(id_str)+8) {
     sprintf(id_str, "%s-%s", PACKAGE, VERSION);
     memcpy(AVI_header+nhb, id_str, strlen(id_str));
   }
   
   nhb += njunk;

   /* Start the movi list */

   OUT4CC ("LIST");
   OUTLONG(movi_len); /* Length of list in bytes */
   OUT4CC ("movi");

   /* Output the header, truncate the file to the number of bytes
      actually written, report an error if someting goes wrong */

   if ( lseek(AVI->fdes,0,SEEK_SET)<0 ||
        avi_write(AVI->fdes,(char *)AVI_header,HEADERBYTES)!=HEADERBYTES ||
	lseek(AVI->fdes,AVI->pos,SEEK_SET)<0)
     {
       AVI_errno = AVI_ERR_CLOSE;
       return -1;
     }

   return 0;
}

/*
  Write the header of an AVI file and close it.
  returns 0 on success, -1 on write error.
*/

static int avi_close_output_file(avi_t *AVI)
{

   int ret, njunk, sampsize, hasIndex, ms_per_frame, frate, idxerror, flag;
   unsigned long movi_len;
   int hdrl_start, strl_start, j;
   unsigned char AVI_header[HEADERBYTES];
   long nhb;

#ifdef INFO_LIST
   long info_len;
//   time_t calptr;
#endif

   /* Calculate length of movi list */

   movi_len = AVI->pos - HEADERBYTES + 4;

   /* Try to ouput the index entries. This may fail e.g. if no space
      is left on device. We will report this as an error, but we still
      try to write the header correctly (so that the file still may be
      readable in the most cases */

   idxerror = 0;
   //   fprintf(stderr, "pos=%lu, index_len=%ld             \n", AVI->pos, AVI->n_idx*16);
   ret = avi_add_chunk(AVI, (unsigned char *)"idx1", (unsigned char *)((void*)AVI->idx), AVI->n_idx*16);
   hasIndex = (ret==0);
   //fprintf(stderr, "pos=%lu, index_len=%d\n", AVI->pos, hasIndex);

   if(ret) {
     idxerror = 1;
     AVI_errno = AVI_ERR_WRITE_INDEX;
   }
   
   /* Calculate Microseconds per frame */

   if(AVI->fps < 0.001) {
     frate=0;
     ms_per_frame=0;
   } else {
     frate = (int) (FRAME_RATE_SCALE*AVI->fps + 0.5);
     ms_per_frame=(int) (1000000/AVI->fps + 0.5);
   }

   /* Prepare the file header */

   nhb = 0;

   /* The RIFF header */

   OUT4CC ("RIFF");
   OUTLONG(AVI->pos - 8);    /* # of bytes to follow */
   OUT4CC ("AVI ");

   /* Start the header list */

   OUT4CC ("LIST");
   OUTLONG(0);        /* Length of list in bytes, don't know yet */
   hdrl_start = nhb;  /* Store start position */
   OUT4CC ("hdrl");

   /* The main AVI header */

   /* The Flags in AVI File header */

#define AVIF_HASINDEX           0x00000010      /* Index at end of file */
#define AVIF_MUSTUSEINDEX       0x00000020
#define AVIF_ISINTERLEAVED      0x00000100
#define AVIF_TRUSTCKTYPE        0x00000800      /* Use CKType to find key frames */
#define AVIF_WASCAPTUREFILE     0x00010000
#define AVIF_COPYRIGHTED        0x00020000

   OUT4CC ("avih");
   OUTLONG(56);                 /* # of bytes to follow */
   OUTLONG(ms_per_frame);       /* Microseconds per frame */
   //ThOe ->0 
   //   OUTLONG(10000000);           /* MaxBytesPerSec, I hope this will never be used */
   OUTLONG(0);
   OUTLONG(0);                  /* PaddingGranularity (whatever that might be) */
                                /* Other sources call it 'reserved' */
   flag = AVIF_ISINTERLEAVED;
   if(hasIndex) flag |= AVIF_HASINDEX;
   if(hasIndex && AVI->must_use_index) flag |= AVIF_MUSTUSEINDEX;
   OUTLONG(flag);               /* Flags */
   OUTLONG(AVI->video_frames);  /* TotalFrames */
   OUTLONG(0);                  /* InitialFrames */

   OUTLONG(AVI->anum+1);
//   if (AVI->track[0].audio_bytes)
//      { OUTLONG(2); }           /* Streams */
//   else
//      { OUTLONG(1); }           /* Streams */

   OUTLONG(0);                  /* SuggestedBufferSize */
   OUTLONG(AVI->width);         /* Width */
   OUTLONG(AVI->height);        /* Height */
                                /* MS calls the following 'reserved': */
   OUTLONG(0);                  /* TimeScale:  Unit used to measure time */
   OUTLONG(0);                  /* DataRate:   Data rate of playback     */
   OUTLONG(0);                  /* StartTime:  Starting time of AVI data */
   OUTLONG(0);                  /* DataLength: Size of AVI data chunk    */


   /* Start the video stream list ---------------------------------- */

   OUT4CC ("LIST");
   OUTLONG(0);        /* Length of list in bytes, don't know yet */
   strl_start = nhb;  /* Store start position */
   OUT4CC ("strl");

   /* The video stream header */

   OUT4CC ("strh");
   OUTLONG(56);                 /* # of bytes to follow */
   OUT4CC ("vids");             /* Type */
   OUT4CC (AVI->compressor);    /* Handler */
   OUTLONG(0);                  /* Flags */
   OUTLONG(0);                  /* Reserved, MS says: wPriority, wLanguage */
   OUTLONG(0);                  /* InitialFrames */
   OUTLONG(FRAME_RATE_SCALE);              /* Scale */
   OUTLONG(frate);              /* Rate: Rate/Scale == samples/second */
   OUTLONG(0);                  /* Start */
   OUTLONG(AVI->video_frames);  /* Length */
   OUTLONG(0);                  /* SuggestedBufferSize */
   OUTLONG(-1);                 /* Quality */
   OUTLONG(0);                  /* SampleSize */
   OUTLONG(0);                  /* Frame */
   OUTLONG(0);                  /* Frame */
   //   OUTLONG(0);                  /* Frame */
   //OUTLONG(0);                  /* Frame */

   /* The video stream format */

   OUT4CC ("strf");
   OUTLONG(40);                 /* # of bytes to follow */
   OUTLONG(40);                 /* Size */
   OUTLONG(AVI->width);         /* Width */
   OUTLONG(AVI->height);        /* Height */
   OUTSHRT(1); OUTSHRT(24);     /* Planes, Count */
   OUT4CC (AVI->compressor);    /* Compression */
   // ThOe (*3)
   OUTLONG(AVI->width*AVI->height*3);  /* SizeImage (in bytes?) */
   OUTLONG(0);                  /* XPelsPerMeter */
   OUTLONG(0);                  /* YPelsPerMeter */
   OUTLONG(0);                  /* ClrUsed: Number of colors used */
   OUTLONG(0);                  /* ClrImportant: Number of colors important */

   /* Finish stream list, i.e. put number of bytes in the list to proper pos */

   long2str(AVI_header+strl_start-4,nhb-strl_start);

   /* Start the audio stream list ---------------------------------- */

   for(j=0; j<AVI->anum; ++j) {
     
     //if (AVI->track[j].a_chans && AVI->track[j].audio_bytes)
       {
	   
	 sampsize = avi_sampsize(AVI, j);
	   
	 OUT4CC ("LIST");
	 OUTLONG(0);        /* Length of list in bytes, don't know yet */
	 strl_start = nhb;  /* Store start position */
	 OUT4CC ("strl");
	   
	 /* The audio stream header */
	 
	 OUT4CC ("strh");
	 OUTLONG(56);            /* # of bytes to follow */
	 OUT4CC ("auds");
	 
	 // -----------
	 // ThOe
	 OUTLONG(0);             /* Format (Optionally) */
	   // -----------
	   
	 OUTLONG(0);             /* Flags */
	 OUTLONG(0);             /* Reserved, MS says: wPriority, wLanguage */
	 OUTLONG(0);             /* InitialFrames */
	   
	 // ThOe /4
	 OUTLONG(sampsize/4);      /* Scale */
	 OUTLONG(1000*AVI->track[j].mp3rate/8);
	 OUTLONG(0);             /* Start */
	 OUTLONG(4*AVI->track[j].audio_bytes/sampsize);   /* Length */
	 OUTLONG(0);             /* SuggestedBufferSize */
	 OUTLONG(-1);            /* Quality */
	   
	 // ThOe /4
	 OUTLONG(sampsize/4);    /* SampleSize */
	   
	 OUTLONG(0);             /* Frame */
	 OUTLONG(0);             /* Frame */
	 //	 OUTLONG(0);             /* Frame */
	 //OUTLONG(0);             /* Frame */
	   
	 /* The audio stream format */
	 
	 OUT4CC ("strf");
	 OUTLONG(16);                   /* # of bytes to follow */
	 OUTSHRT(AVI->track[j].a_fmt);           /* Format */
	 OUTSHRT(AVI->track[j].a_chans);         /* Number of channels */
	 OUTLONG(AVI->track[j].a_rate);          /* SamplesPerSec */
	 // ThOe
	 OUTLONG(1000*AVI->track[j].mp3rate/8);
	 //ThOe (/4)
	 
	 OUTSHRT(sampsize/4);           /* BlockAlign */
	 
	 
	 OUTSHRT(AVI->track[j].a_bits);          /* BitsPerSample */
	 
	 /* Finish stream list, i.e. put number of bytes in the list to proper pos */
       }
       long2str(AVI_header+strl_start-4,nhb-strl_start);
   }
   
   /* Finish header list */
   
   long2str(AVI_header+hdrl_start-4,nhb-hdrl_start);


   // add INFO list --- (0.6.0pre4)

#ifdef INFO_LIST
   OUT4CC ("LIST");
   
   //FIXME
   info_len = MAX_INFO_STRLEN + 12;
   OUTLONG(info_len);
   OUT4CC ("INFO");

//   OUT4CC ("INAM");
//   OUTLONG(MAX_INFO_STRLEN);

//   sprintf(id_str, "\t");
//   memset(AVI_header+nhb, 0, MAX_INFO_STRLEN);
//   memcpy(AVI_header+nhb, id_str, strlen(id_str));
//   nhb += MAX_INFO_STRLEN;

   OUT4CC ("ISFT");
   OUTLONG(MAX_INFO_STRLEN);

   sprintf(id_str, "%s-%s", PACKAGE, VERSION);
   memset(AVI_header+nhb, 0, MAX_INFO_STRLEN);
   memcpy(AVI_header+nhb, id_str, strlen(id_str));
   nhb += MAX_INFO_STRLEN;

//   OUT4CC ("ICMT");
//   OUTLONG(MAX_INFO_STRLEN);

//   calptr=time(NULL); 
//   sprintf(id_str, "\t%s %s", ctime(&calptr), "");
//   memset(AVI_header+nhb, 0, MAX_INFO_STRLEN);
//   memcpy(AVI_header+nhb, id_str, 25);
//   nhb += MAX_INFO_STRLEN;
#endif

   // ----------------------------
   
   /* Calculate the needed amount of junk bytes, output junk */
   
   njunk = HEADERBYTES - nhb - 8 - 12;
   
   /* Safety first: if njunk <= 0, somebody has played with
      HEADERBYTES without knowing what (s)he did.
      This is a fatal error */
   
   if(njunk<=0)
   {
      fprintf(stderr,"AVI_close_output_file: # of header bytes too small\n");
      exit(1);
   }

   OUT4CC ("JUNK");
   OUTLONG(njunk);
   memset(AVI_header+nhb,0,njunk);
   
   nhb += njunk;

   /* Start the movi list */

   OUT4CC ("LIST");
   OUTLONG(movi_len); /* Length of list in bytes */
   OUT4CC ("movi");

   /* Output the header, truncate the file to the number of bytes
      actually written, report an error if someting goes wrong */

   if ( lseek(AVI->fdes,0,SEEK_SET)<0 ||
        avi_write(AVI->fdes,(char *)AVI_header,HEADERBYTES)!=HEADERBYTES 
        //|| ftruncate(AVI->fdes,AVI->pos)<0 
        )
   {
      AVI_errno = AVI_ERR_CLOSE;
      return -1;
   }

   if(idxerror) return -1;

   return 0;
}

/*
   AVI_write_data:
   Add video or audio data to the file;

   Return values:
    0    No error;
   -1    Error, AVI_errno is set appropriatly;

*/

static int avi_write_data(avi_t *AVI, char *data, unsigned long length, int audio, int keyframe)
{
   int n;

   unsigned char astr[5];

   /* Check for maximum file length */
   
   if ( (AVI->pos + 8 + length + 8 + (AVI->n_idx+1)*16) > AVI_MAX_LEN ) {
     AVI_errno = AVI_ERR_SIZELIM;
     return -1;
   }
   
   /* Add index entry */

   //set tag for current audio track
   sprintf((char *)astr, "0%1dwb", AVI->aptr+1);

   if(audio)
     n = avi_add_index_entry(AVI,astr,0x00,AVI->pos,length);
   else
     n = avi_add_index_entry(AVI,(unsigned char *) "00db",((keyframe)?0x10:0x0),AVI->pos,length);
   
   if(n) return -1;
   
   /* Output tag and data */
   
   if(audio)
     n = avi_add_chunk(AVI,(unsigned char *) astr, (unsigned char *)data,length);
   else
     n = avi_add_chunk(AVI,(unsigned char *)"00db",(unsigned char *)data,length);
   
   if (n) return -1;
   
   return 0;
}

int AVI_write_frame(avi_t *AVI, char *data, long bytes, int keyframe)
{
  unsigned long pos;
  
  if(AVI->mode==AVI_MODE_READ) { AVI_errno = AVI_ERR_NOT_PERM; return -1; }
  
  pos = AVI->pos;

  if(avi_write_data(AVI,data,bytes,0,keyframe)) return -1;
   
  AVI->last_pos = pos;
  AVI->last_len = bytes;
  AVI->video_frames++;
  return 0;
}

int AVI_dup_frame(avi_t *AVI)
{
   if(AVI->mode==AVI_MODE_READ) { AVI_errno = AVI_ERR_NOT_PERM; return -1; }

   if(AVI->last_pos==0) return 0; /* No previous real frame */
   if(avi_add_index_entry(AVI,(unsigned char *)"00db",0x10,AVI->last_pos,AVI->last_len)) return -1;
   AVI->video_frames++;
   AVI->must_use_index = 1;
   return 0;
}

int AVI_write_audio(avi_t *AVI, char *data, long bytes)
{
   if(AVI->mode==AVI_MODE_READ) { AVI_errno = AVI_ERR_NOT_PERM; return -1; }

   if( avi_write_data(AVI,data,bytes,1,0) ) return -1;
   AVI->track[AVI->aptr].audio_bytes += bytes;
   return 0;
}


int AVI_append_audio(avi_t *AVI, char *data, long bytes)
{

  long i, length, pos;
  unsigned char c[4];

  if(AVI->mode==AVI_MODE_READ) { AVI_errno = AVI_ERR_NOT_PERM; return -1; }
  
  // update last index entry:
  
  --AVI->n_idx;
  length = str2ulong(AVI->idx[AVI->n_idx]+12);
  pos    = str2ulong(AVI->idx[AVI->n_idx]+8);

  //update;
  long2str(AVI->idx[AVI->n_idx]+12,length+bytes);   

  ++AVI->n_idx;

  AVI->track[AVI->aptr].audio_bytes += bytes;

  //update chunk header
  lseek(AVI->fdes, pos+4, SEEK_SET);
  long2str(c, length+bytes);     
  avi_write(AVI->fdes,(char *) c, 4);

  lseek(AVI->fdes, pos+8+length, SEEK_SET);

  i=PAD_EVEN(length + bytes);

  bytes = i - length;
  avi_write(AVI->fdes, data, bytes);
  AVI->pos = pos + 8 + i;

  return 0;
}


long AVI_bytes_remain(avi_t *AVI)
{
   if(AVI->mode==AVI_MODE_READ) return 0;

   return ( AVI_MAX_LEN - (AVI->pos + 8 + 16*AVI->n_idx));
}

long AVI_bytes_written(avi_t *AVI)
{
   if(AVI->mode==AVI_MODE_READ) return 0;

   return (AVI->pos + 8 + 16*AVI->n_idx);
}

int AVI_set_audio_track(avi_t *AVI, int track)
{
  
  if(track < 0 || track + 1 > AVI->anum) return(-1);

  //this info is not written to file anyway
  AVI->aptr=track;
  return 0;
}

int AVI_get_audio_track(avi_t *AVI)
{
    return(AVI->aptr);
}


/*******************************************************************
 *                                                                 *
 *    Utilities for reading video and audio from an AVI File       *
 *                                                                 *
 *******************************************************************/

int AVI_close(avi_t *AVI)
{
   int ret;

   /* If the file was open for writing, the header and index still have
      to be written */

   if(AVI->mode == AVI_MODE_WRITE)
      ret = avi_close_output_file(AVI);
   else
      ret = 0;

   /* Even if there happened an error, we first clean up */

   close(AVI->fdes);
   if(AVI->idx) free(AVI->idx);
   if(AVI->video_index) free(AVI->video_index);
   //FIXME
   //if(AVI->audio_index) free(AVI->audio_index);
   free(AVI);

   return ret;
}


#define ERR_EXIT(x) \
{ \
   AVI_close(AVI); \
   AVI_errno = x; \
   return 0; \
}

avi_t *AVI_open_input_file(char *filename, int getIndex)
{
  avi_t *AVI=NULL;
  
  /* Create avi_t structure */
  
  AVI = (avi_t *) malloc(sizeof(avi_t));
  if(AVI==NULL)
    {
      AVI_errno = AVI_ERR_NO_MEM;
      return 0;
    }
  memset((void *)AVI,0,sizeof(avi_t));
  
  AVI->mode = AVI_MODE_READ; /* open for reading */
  
  /* Open the file */
  
  AVI->fdes = open(filename,O_RDONLY|O_BINARY);
  if(AVI->fdes < 0)
    {
      AVI_errno = AVI_ERR_OPEN;
      free(AVI);
      return 0;
    }
  
  avi_parse_input_file(AVI, getIndex);

  AVI->aptr=0; //reset  

  return AVI;
}

avi_t *AVI_open_fd(int fd, int getIndex)
{
  avi_t *AVI=NULL;
  
  /* Create avi_t structure */
  
  AVI = (avi_t *) malloc(sizeof(avi_t));
  if(AVI==NULL)
    {
      AVI_errno = AVI_ERR_NO_MEM;
      return 0;
    }
  memset((void *)AVI,0,sizeof(avi_t));
  
  AVI->mode = AVI_MODE_READ; /* open for reading */
  
  // file alread open
  AVI->fdes = fd;
  
  avi_parse_input_file(AVI, getIndex);

  AVI->aptr=0; //reset
  
  return AVI;
}

int avi_parse_input_file(avi_t *AVI, int getIndex)
{
  long i, n, rate, scale, idx_type;
  unsigned char *hdrl_data;
  long header_offset=0, hdrl_len=0;
  long nvi, nai[AVI_MAX_TRACKS], ioff;
  long tot[AVI_MAX_TRACKS];
  int j;
  int lasttag = 0;
  int vids_strh_seen = 0;
  int vids_strf_seen = 0;
  int auds_strh_seen = 0;
  //  int auds_strf_seen = 0;
  int num_stream = 0;
  char data[256];
  
  /* Read first 12 bytes and check that this is an AVI file */

   if( avi_read(AVI->fdes,data,12) != 12 ) ERR_EXIT(AVI_ERR_READ)

   if( strncasecmp(data  ,"RIFF",4) !=0 ||
       strncasecmp(data+8,"AVI ",4) !=0 ) ERR_EXIT(AVI_ERR_NO_AVI)

   /* Go through the AVI file and extract the header list,
      the start position of the 'movi' list and an optionally
      present idx1 tag */

   hdrl_data = 0;

   while(1)
   {
      if( avi_read(AVI->fdes,data,8) != 8 ) break; /* We assume it's EOF */

      n = str2ulong((unsigned char *) data+4);
      n = PAD_EVEN(n);

      if(strncasecmp(data,"LIST",4) == 0)
      {
         if( avi_read(AVI->fdes,data,4) != 4 ) ERR_EXIT(AVI_ERR_READ)
         n -= 4;
         if(strncasecmp(data,"hdrl",4) == 0)
         {
            hdrl_len = n;
            hdrl_data = (unsigned char *) malloc(n);
            if(hdrl_data==0) ERR_EXIT(AVI_ERR_NO_MEM);
				 
	    // offset of header
	    
	    header_offset = lseek(AVI->fdes,0,SEEK_CUR);
				 
            if( avi_read(AVI->fdes,(char *)hdrl_data,n) != n ) ERR_EXIT(AVI_ERR_READ)
         }
         else if(strncasecmp(data,"movi",4) == 0)
         {
            AVI->movi_start = lseek(AVI->fdes,0,SEEK_CUR);
            lseek(AVI->fdes,n,SEEK_CUR);
         }
         else
            lseek(AVI->fdes,n,SEEK_CUR);
      }
      else if(strncasecmp(data,"idx1",4) == 0)
      {
         /* n must be a multiple of 16, but the reading does not
            break if this is not the case */

         AVI->n_idx = AVI->max_idx = n/16;
         AVI->idx = (unsigned  char((*)[16]) ) malloc(n);
         if(AVI->idx==0) ERR_EXIT(AVI_ERR_NO_MEM)
         if(avi_read(AVI->fdes, (char *) AVI->idx, n) != n ) ERR_EXIT(AVI_ERR_READ)
      }
      else
         lseek(AVI->fdes,n,SEEK_CUR);
   }

   if(!hdrl_data      ) ERR_EXIT(AVI_ERR_NO_HDRL)
   if(!AVI->movi_start) ERR_EXIT(AVI_ERR_NO_MOVI)

   /* Interpret the header list */

   for(i=0;i<hdrl_len;)
   {
      /* List tags are completly ignored */

      if(strncasecmp((char *) hdrl_data+i, "LIST",4)==0) { i+= 12; continue; }

      n = str2ulong(hdrl_data+i+4);
      n = PAD_EVEN(n);

      /* Interpret the tag and its args */

      if(strncasecmp((char *)hdrl_data+i,"strh",4)==0)
      {
         i += 8;
         if(strncasecmp((char *)hdrl_data+i,"vids",4) == 0 && !vids_strh_seen)
         {
            memcpy(AVI->compressor,hdrl_data+i+4,4);
            AVI->compressor[4] = 0;

	    // ThOe
	    AVI->v_codech_off = header_offset + i+4;

            scale = str2ulong((unsigned char *)hdrl_data+i+20);
            rate  = str2ulong(hdrl_data+i+24);
            if(scale!=0) AVI->fps = (double)rate/(double)scale;
            AVI->video_frames = str2ulong(hdrl_data+i+32);
            AVI->video_strn = num_stream;
	    AVI->max_len = 0;
            vids_strh_seen = 1;
            lasttag = 1; /* vids */
         }
         else if (strncasecmp ((char *) hdrl_data+i,"auds",4) ==0 && ! auds_strh_seen)
         {

	   //inc audio tracks
	   AVI->aptr=AVI->anum;
	   ++AVI->anum;
	   
	   if(AVI->anum > AVI_MAX_TRACKS) {
	     fprintf(stderr, "error - only %d audio tracks supported\n", AVI_MAX_TRACKS);
	     return(-1);
	   }
	   
	   AVI->track[AVI->aptr].audio_bytes = str2ulong(hdrl_data+i+32)*avi_sampsize(AVI, 0);
	   AVI->track[AVI->aptr].audio_strn = num_stream;
	   //	   auds_strh_seen = 1;
	   lasttag = 2; /* auds */
	   
	   // ThOe
	   AVI->track[AVI->aptr].a_codech_off = header_offset + i;
	   
         }
         else
            lasttag = 0;
         num_stream++;
      }
      else if(strncasecmp((char *) hdrl_data+i,"strf",4)==0)
      {
         i += 8;
         if(lasttag == 1)
         {
            AVI->width  = str2ulong(hdrl_data+i+4);
            AVI->height = str2ulong(hdrl_data+i+8);
            vids_strf_seen = 1;
	    //ThOe
	    AVI->v_codecf_off = header_offset + i+16;

	    memcpy(AVI->compressor2, hdrl_data+i+16, 4);
            AVI->compressor2[4] = 0;

         }
         else if(lasttag == 2)
         {
            AVI->track[AVI->aptr].a_fmt   = str2ushort(hdrl_data+i  );

	    //ThOe
	    AVI->track[AVI->aptr].a_codecf_off = header_offset + i;
	    
            AVI->track[AVI->aptr].a_chans = str2ushort(hdrl_data+i+2);
            AVI->track[AVI->aptr].a_rate  = str2ulong (hdrl_data+i+4);
	    //ThOe: read mp3bitrate
	    AVI->track[AVI->aptr].mp3rate = 8*str2ulong(hdrl_data+i+8)/1000;
	    //:ThOe
            AVI->track[AVI->aptr].a_bits  = str2ushort(hdrl_data+i+14);
	    //            auds_strf_seen = 1;
         }
         lasttag = 0;
      }
      else
      {
         i += 8;
         lasttag = 0;
      }

      i += n;
   }

   free(hdrl_data);

   if(!vids_strh_seen || !vids_strf_seen) ERR_EXIT(AVI_ERR_NO_VIDS)

   AVI->video_tag[0] = AVI->video_strn/10 + '0';
   AVI->video_tag[1] = AVI->video_strn%10 + '0';
   AVI->video_tag[2] = 'd';
   AVI->video_tag[3] = 'b';

   /* Audio tag is set to "99wb" if no audio present */
   if(!AVI->track[0].a_chans) AVI->track[0].audio_strn = 99;

   for(j=0; j<AVI->anum; ++j) {
     AVI->track[j].audio_tag[0] = (j+1)/10 + '0';
     AVI->track[j].audio_tag[1] = (j+1)%10 + '0';
     AVI->track[j].audio_tag[2] = 'w';
     AVI->track[j].audio_tag[3] = 'b';
   }

   lseek(AVI->fdes,AVI->movi_start,SEEK_SET);

   /* get index if wanted */

   if(!getIndex) return(0);

   /* if the file has an idx1, check if this is relative
      to the start of the file or to the start of the movi list */

   idx_type = 0;

   if(AVI->idx)
   {
      long pos, len;

      /* Search the first videoframe in the idx1 and look where
         it is in the file */

      for(i=0;i<AVI->n_idx;i++)
         if( strncasecmp((char *) AVI->idx[i],(char *) AVI->video_tag,3)==0 ) break;
      if(i>=AVI->n_idx) ERR_EXIT(AVI_ERR_NO_VIDS)

      pos = str2ulong(AVI->idx[i]+ 8);
      len = str2ulong(AVI->idx[i]+12);

      lseek(AVI->fdes,pos,SEEK_SET);
      if(avi_read(AVI->fdes,data,8)!=8) ERR_EXIT(AVI_ERR_READ)
      if( strncasecmp((char *)data,(char *)AVI->idx[i],4)==0 && 
      str2ulong((unsigned char *)data+4)==len )
      {
         idx_type = 1; /* Index from start of file */
      }
      else
      {
         lseek(AVI->fdes,pos+AVI->movi_start-4,SEEK_SET);
         if(avi_read(AVI->fdes,data,8)!=8) ERR_EXIT(AVI_ERR_READ)
         if( strncasecmp((char *)data,(char *)AVI->idx[i],4)==0 && str2ulong((unsigned char *)data+4)==len )
         {
            idx_type = 2; /* Index from start of movi list */
         }
      }
      /* idx_type remains 0 if neither of the two tests above succeeds */
   }

   if(idx_type == 0)
   {
      /* we must search through the file to get the index */

      lseek(AVI->fdes, AVI->movi_start, SEEK_SET);

      AVI->n_idx = 0;

      while(1)
      {
         if( avi_read(AVI->fdes,data,8) != 8 ) break;
         n = str2ulong((unsigned char *)data+4);

         /* The movi list may contain sub-lists, ignore them */

         if(strncasecmp(data,"LIST",4)==0)
         {
            lseek(AVI->fdes,4,SEEK_CUR);
            continue;
         }

         /* Check if we got a tag ##db, ##dc or ##wb */
	 
         if( ( (data[2]=='d' || data[2]=='D') &&
               (data[3]=='b' || data[3]=='B' || data[3]=='c' || data[3]=='C') )
	     || ( (data[2]=='w' || data[2]=='W') &&
		  (data[3]=='b' || data[3]=='B') ) )
	   {
	   avi_add_index_entry(AVI,(unsigned char *) data,0,lseek(AVI->fdes,0,SEEK_CUR)-8,n);
         }
	 
         lseek(AVI->fdes,PAD_EVEN(n),SEEK_CUR);
      }
      idx_type = 1;
   }

   /* Now generate the video index and audio index arrays */

   nvi = 0;
   for(j=0; j<AVI->anum; ++j) nai[j] = 0;

   for(i=0;i<AVI->n_idx;i++) {
     
     if(strncasecmp((char *)AVI->idx[i],(char *) AVI->video_tag,3) == 0) nvi++;
     
     for(j=0; j<AVI->anum; ++j) if(strncasecmp((char *)AVI->idx[i], AVI->track[j].audio_tag,4) == 0) nai[j]++;
   }
   
   AVI->video_frames = nvi;
   for(j=0; j<AVI->anum; ++j) AVI->track[j].audio_chunks = nai[j];

//   fprintf(stderr, "chunks = %ld %d %s\n", AVI->track[0].audio_chunks, AVI->anum, AVI->track[0].audio_tag);

   if(AVI->video_frames==0) ERR_EXIT(AVI_ERR_NO_VIDS);
   AVI->video_index = (video_index_entry *) malloc(nvi*sizeof(video_index_entry));
   if(AVI->video_index==0) ERR_EXIT(AVI_ERR_NO_MEM);
   
   for(j=0; j<AVI->anum; ++j) {
       if(AVI->track[j].audio_chunks) {
	   AVI->track[j].audio_index = (audio_index_entry *) malloc(nai[j]*sizeof(audio_index_entry));
	   if(AVI->track[j].audio_index==0) ERR_EXIT(AVI_ERR_NO_MEM);
       }
   }   
   
   nvi = 0;
   for(j=0; j<AVI->anum; ++j) nai[j] = tot[j] = 0;
   
   ioff = idx_type == 1 ? 8 : AVI->movi_start+4;
   
   for(i=0;i<AVI->n_idx;i++) {

     //video
     if(strncasecmp((char *)AVI->idx[i],(char *)AVI->video_tag,3) == 0) {
       AVI->video_index[nvi].key = str2ulong(AVI->idx[i]+ 4);
       AVI->video_index[nvi].pos = str2ulong(AVI->idx[i]+ 8)+ioff;
       AVI->video_index[nvi].len = str2ulong(AVI->idx[i]+12);
       nvi++;
     }
     
     //audio
     for(j=0; j<AVI->anum; ++j) {
	 
       if(strncasecmp((char *)AVI->idx[i],AVI->track[j].audio_tag,4) == 0) {
	 AVI->track[j].audio_index[nai[j]].pos = str2ulong(AVI->idx[i]+ 8)+ioff;
	 AVI->track[j].audio_index[nai[j]].len = str2ulong(AVI->idx[i]+12);
	 AVI->track[j].audio_index[nai[j]].tot = tot[j];
	 tot[j] += AVI->track[j].audio_index[nai[j]].len;
	 nai[j]++;
       }
     }
   }
   
   
   for(j=0; j<AVI->anum; ++j) AVI->track[j].audio_bytes = tot[j];
   
   /* Reposition the file */
   
   lseek(AVI->fdes,AVI->movi_start,SEEK_SET);
   AVI->video_pos = 0;

   return(0);
}

long AVI_video_frames(avi_t *AVI)
{
   return AVI->video_frames;
}
int  AVI_video_width(avi_t *AVI)
{
   return AVI->width;
}
int  AVI_video_height(avi_t *AVI)
{
   return AVI->height;
}
double AVI_frame_rate(avi_t *AVI)
{
   return AVI->fps;
}
char* AVI_video_compressor(avi_t *AVI)
{
   return AVI->compressor2;
}

long AVI_max_video_chunk(avi_t *AVI)
{
   return AVI->max_len; 
}

int AVI_audio_tracks(avi_t *AVI)
{
    return(AVI->anum);
}

int AVI_audio_channels(avi_t *AVI)
{
   return AVI->track[AVI->aptr].a_chans;
}

long AVI_audio_mp3rate(avi_t *AVI)
{
   return AVI->track[AVI->aptr].mp3rate;
}

int AVI_audio_bits(avi_t *AVI)
{
   return AVI->track[AVI->aptr].a_bits;
}

int AVI_audio_format(avi_t *AVI)
{
   return AVI->track[AVI->aptr].a_fmt;
}

long AVI_audio_rate(avi_t *AVI)
{
   return AVI->track[AVI->aptr].a_rate;
}

long AVI_audio_bytes(avi_t *AVI)
{
   return AVI->track[AVI->aptr].audio_bytes;
}

long AVI_audio_chunks(avi_t *AVI)
{
   return AVI->track[AVI->aptr].audio_chunks;
}

long AVI_audio_codech_offset(avi_t *AVI)
{
   return AVI->track[AVI->aptr].a_codech_off;
}

long AVI_audio_codecf_offset(avi_t *AVI)
{
   return AVI->track[AVI->aptr].a_codecf_off;
}

long  AVI_video_codech_offset(avi_t *AVI)
{
    return AVI->v_codech_off;
}

long  AVI_video_codecf_offset(avi_t *AVI)
{
    return AVI->v_codecf_off;
}

long AVI_frame_size(avi_t *AVI, long frame)
{
   if(AVI->mode==AVI_MODE_WRITE) { AVI_errno = AVI_ERR_NOT_PERM; return -1; }
   if(!AVI->video_index)         { AVI_errno = AVI_ERR_NO_IDX;   return -1; }

   if(frame < 0 || frame >= AVI->video_frames) return 0;
   return(AVI->video_index[frame].len);
}

long AVI_audio_size(avi_t *AVI, long frame)
{
  if(AVI->mode==AVI_MODE_WRITE) { AVI_errno = AVI_ERR_NOT_PERM; return -1; }
  if(!AVI->track[AVI->aptr].audio_index)         { AVI_errno = AVI_ERR_NO_IDX;   return -1; }
  
  if(frame < 0 || frame >= AVI->track[AVI->aptr].audio_chunks) return 0;
  return(AVI->track[AVI->aptr].audio_index[frame].len);
}

long AVI_get_video_position(avi_t *AVI, long frame)
{
   if(AVI->mode==AVI_MODE_WRITE) { AVI_errno = AVI_ERR_NOT_PERM; return -1; }
   if(!AVI->video_index)         { AVI_errno = AVI_ERR_NO_IDX;   return -1; }

   if(frame < 0 || frame >= AVI->video_frames) return 0;
   return(AVI->video_index[frame].pos);
}


int AVI_seek_start(avi_t *AVI)
{
   if(AVI->mode==AVI_MODE_WRITE) { AVI_errno = AVI_ERR_NOT_PERM; return -1; }

   lseek(AVI->fdes,AVI->movi_start,SEEK_SET);
   AVI->video_pos = 0;
   return 0;
}

int AVI_set_video_position(avi_t *AVI, long frame)
{
   if(AVI->mode==AVI_MODE_WRITE) { AVI_errno = AVI_ERR_NOT_PERM; return -1; }
   if(!AVI->video_index)         { AVI_errno = AVI_ERR_NO_IDX;   return -1; }

   if (frame < 0 ) frame = 0;
   AVI->video_pos = frame;
   return 0;
}

int AVI_set_audio_bitrate(avi_t *AVI, long bitrate)
{
   if(AVI->mode==AVI_MODE_READ) { AVI_errno = AVI_ERR_NOT_PERM; return -1; }

   AVI->track[AVI->aptr].mp3rate = bitrate;
   return 0;
}
      

long AVI_read_frame(avi_t *AVI, char *vidbuf, int *keyframe)
{
   long n;
   if(AVI->mode==AVI_MODE_WRITE) { AVI_errno = AVI_ERR_NOT_PERM; return -1; }
   if(!AVI->video_index)         { AVI_errno = AVI_ERR_NO_IDX;   return -1; }
   if(AVI->video_pos < 0 || AVI->video_pos >= AVI->video_frames) return -1;
   n = AVI->video_index[AVI->video_pos].len;
   *keyframe = (AVI->video_index[AVI->video_pos].key==0x10) ? 1:0;
   lseek(AVI->fdes, AVI->video_index[AVI->video_pos].pos, SEEK_SET);
   if (avi_read(AVI->fdes,vidbuf,n) != n)
   {
      AVI_errno = AVI_ERR_READ;
      return -1;
   }
   AVI->video_pos++;
   return n;
}

int AVI_set_audio_position(avi_t *AVI, long byte)
{
   long n0, n1, n;

   if(AVI->mode==AVI_MODE_WRITE) { AVI_errno = AVI_ERR_NOT_PERM; return -1; }
   if(!AVI->track[AVI->aptr].audio_index)         { AVI_errno = AVI_ERR_NO_IDX;   return -1; }

   if(byte < 0) byte = 0;

   /* Binary search in the audio chunks */

   n0 = 0;
   n1 = AVI->track[AVI->aptr].audio_chunks;

   while(n0<n1-1)
   {
      n = (n0+n1)/2;
      if(AVI->track[AVI->aptr].audio_index[n].tot>byte)
         n1 = n;
      else
         n0 = n;
   }

   AVI->track[AVI->aptr].audio_posc = n0;
   AVI->track[AVI->aptr].audio_posb = byte - AVI->track[AVI->aptr].audio_index[n0].tot;

   return 0;
}

long AVI_read_audio(avi_t *AVI, char *audbuf, long bytes)
{
   long nr, pos, left, todo;

   if(AVI->mode==AVI_MODE_WRITE) { AVI_errno = AVI_ERR_NOT_PERM; return -1; }
   if(!AVI->track[AVI->aptr].audio_index)         { AVI_errno = AVI_ERR_NO_IDX;   return -1; }

   nr = 0; /* total number of bytes read */

   while(bytes>0)
   {
      left = AVI->track[AVI->aptr].audio_index[AVI->track[AVI->aptr].audio_posc].len - AVI->track[AVI->aptr].audio_posb;
      if(left==0)
      {
         if(AVI->track[AVI->aptr].audio_posc>=AVI->track[AVI->aptr].audio_chunks-1) return nr;
         AVI->track[AVI->aptr].audio_posc++;
         AVI->track[AVI->aptr].audio_posb = 0;
         continue;
      }
      if(bytes<left)
         todo = bytes;
      else
         todo = left;
      pos = AVI->track[AVI->aptr].audio_index[AVI->track[AVI->aptr].audio_posc].pos + AVI->track[AVI->aptr].audio_posb;
      lseek(AVI->fdes, pos, SEEK_SET);
      if (avi_read(AVI->fdes,audbuf+nr,todo) != todo)
      {
         AVI_errno = AVI_ERR_READ;
         return -1;
      }
      bytes -= todo;
      nr    += todo;
      AVI->track[AVI->aptr].audio_posb += todo;
   }

   return nr;
}

/* AVI_read_data: Special routine for reading the next audio or video chunk
                  without having an index of the file. */

int AVI_read_data(avi_t *AVI, char *vidbuf, long max_vidbuf,
                              char *audbuf, long max_audbuf,
                              long *len)
{

/*
 * Return codes:
 *
 *    1 = video data read
 *    2 = audio data read
 *    0 = reached EOF
 *   -1 = video buffer too small
 *   -2 = audio buffer too small
 */

   int n;
   char data[8];
 
   if(AVI->mode==AVI_MODE_WRITE) return 0;

   while(1)
   {
      /* Read tag and length */

      if( avi_read(AVI->fdes,data,8) != 8 ) return 0;

      /* if we got a list tag, ignore it */

      if(strncasecmp(data,"LIST",4) == 0)
      {
         lseek(AVI->fdes,4,SEEK_CUR);
         continue;
      }

      n = PAD_EVEN(str2ulong((unsigned char *)data+4));

      if(strncasecmp(data,AVI->video_tag,3) == 0)
      {
         *len = n;
         AVI->video_pos++;
         if(n>max_vidbuf)
         {
            lseek(AVI->fdes,n,SEEK_CUR);
            return -1;
         }
         if(avi_read(AVI->fdes,vidbuf,n) != n ) return 0;
         return 1;
      }
      else if(strncasecmp(data,AVI->track[AVI->aptr].audio_tag,4) == 0)
      {
         *len = n;
         if(n>max_audbuf)
         {
            lseek(AVI->fdes,n,SEEK_CUR);
            return -2;
         }
         if(avi_read(AVI->fdes,audbuf,n) != n ) return 0;
         return 2;
         break;
      }
      else
         if(lseek(AVI->fdes,n,SEEK_CUR)<0)  return 0;
   }
}

/* AVI_print_error: Print most recent error (similar to perror) */

char *(avi_errors[]) =
{
  /*  0 */ (char *) "avilib - No Error",
  /*  1 */ (char *) "avilib - AVI file size limit reached",
  /*  2 */ (char *) "avilib - Error opening AVI file",
  /*  3 */ (char *) "avilib - Error reading from AVI file",
  /*  4 */ (char *) "avilib - Error writing to AVI file",
  /*  5 */ (char *) "avilib - Error writing index (file may still be useable)",
  /*  6 */ (char *) "avilib - Error closing AVI file",
  /*  7 */ (char *) "avilib - Operation (read/write) not permitted",
  /*  8 */ (char *) "avilib - Out of memory (malloc failed)",
  /*  9 */ (char *) "avilib - Not an AVI file",
  /* 10 */ (char *) "avilib - AVI file has no header list (corrupted?)",
  /* 11 */ (char *) "avilib - AVI file has no MOVI list (corrupted?)",
  /* 12 */ (char *) "avilib - AVI file has no video data",
  /* 13 */ (char *) "avilib - operation needs an index",
  /* 14 */ (char *) "avilib - Unkown Error"
};
static int num_avi_errors = sizeof(avi_errors)/sizeof(char*);

static char error_string[4096];

void AVI_print_error(char *str)
{
   int aerrno;

   aerrno = (AVI_errno>=0 && AVI_errno<num_avi_errors) ? AVI_errno : num_avi_errors-1;

   fprintf(stderr,"%s: %s\n",str,avi_errors[aerrno]);

   /* for the following errors, perror should report a more detailed reason: */

   if(AVI_errno == AVI_ERR_OPEN ||
      AVI_errno == AVI_ERR_READ ||
      AVI_errno == AVI_ERR_WRITE ||
      AVI_errno == AVI_ERR_WRITE_INDEX ||
      AVI_errno == AVI_ERR_CLOSE )
   {
      perror("REASON");
   }
}

char *AVI_strerror()
{
   int aerrno;

   aerrno = (AVI_errno>=0 && AVI_errno<num_avi_errors) ? AVI_errno : num_avi_errors-1;

   if(AVI_errno == AVI_ERR_OPEN ||
      AVI_errno == AVI_ERR_READ ||
      AVI_errno == AVI_ERR_WRITE ||
      AVI_errno == AVI_ERR_WRITE_INDEX ||
      AVI_errno == AVI_ERR_CLOSE )
   {
      sprintf(error_string,"%s - %s",avi_errors[aerrno],strerror(errno));
      return error_string;
   }
   else
   {
      return avi_errors[aerrno];
   }
}

uint64_t AVI_max_size()
{
  return((uint64_t) AVI_MAX_LEN);
}

#ifdef __cplusplus
}
#endif

// #ifdef __cplusplus
// extern "C" {
// #endif

//===============================================================================================================================================================================================================
//	DEFINE / INCLUDE
//===============================================================================================================================================================================================================
// #include "avimod.h"

//===============================================================================================================================================================================================================
//	FUNCTIONS
//===============================================================================================================================================================================================================

// Flips the specified image and crops it to the specified dimensions
// If scaled == true, all values are scaled to the range [0.0, 1.0
fp* chop_flip_image(	char *image, 
								int height, 
								int width, 
								int cropped,
								int scaled,
								int converted) {

	// fixed dimensions for cropping or not cropping, square vertices starting from initial point in top left corner going down and right
	int top;
	int bottom;
	int left;
	int right;
	if(cropped==1){
		top = 0;
		bottom = 0;
		left = 0;
		right = 0;
	}
	else{
		top = 0;
		bottom = height - 1;
		left = 0;
		right = width - 1;
	}

	// dimensions of new cropped image
	int height_new = bottom - top + 1;
	int width_new = right - left + 1;

	// counters
	int i, j;

	// allocate memory for cropped/flipped frame
	fp* result = (fp *) malloc(height_new * width_new * sizeof(fp));

	// crop/flip and scale frame
	fp temp;
	if (scaled) {
		fp scale = 1.0 / 255.0;
		for(i = 0; i <height_new; i++){				// rows
			for(j = 0; j <width_new; j++){			// colums
				temp = (fp) image[((height - 1 - (i + top)) * width) + (j + left)] * scale;
				if(temp<0){
					result[i*width_new+j] = temp + 256;
				}
				else{
					result[i*width_new+j] = temp;
				}
			}
		}
	} else {
		for(i = 0; i <height_new; i++){				// rows
			for(j = 0; j <width_new; j++){			// colums
				temp = (fp) image[((height - 1 - (i + top)) * width) + (j + left)] ;
				if(temp<0){
					result[i*width_new+j] = temp + 256;
				}
				else{
					result[i*width_new+j] = temp;
				}
			}
		}
	}

// convert storage method (from row-major to column-major)
	fp* result_converted = (fp *) malloc(height_new * width_new * sizeof(fp));
	if(converted==1){
		for(i = 0; i <width_new; i++){				// rows
			for(j = 0; j <height_new; j++){			// colums
				result_converted[i*height_new+j] = result[j*width_new+i];
			}
		}
	}
	else{
		result_converted = result;
	}
	free(result);

	// return
	return result_converted;
}

// Returns the specified frame from the specified video file
// If cropped == true, the frame is cropped to pre-determined dimensions
//  (hardcoded to the boundaries of the blood vessel in the test video)
// If scaled == true, all values are scaled to the range [0.0, 1.0]
fp* get_frame(	avi_t* cell_file, 
						int frame_num, 
						int cropped, 
						int scaled,
						int converted) {

	// variable
	int dummy;
	int width = AVI_video_width(cell_file);
	int height = AVI_video_height(cell_file);
	int status;

	// There are 600 frames in this file (i.e. frame_num = 600 causes an error)
	AVI_set_video_position(cell_file, frame_num);

	//Read in the frame from the AVI
	char* image_buf = (char*) malloc(width * height * sizeof(char));
	status = AVI_read_frame(cell_file, image_buf, &dummy);
	if(status == -1) {
		AVI_print_error((char*) "Error with AVI_read_frame");
		exit(-1);
	}

	// The image is read in upside-down, so we need to flip it
	fp* image_chopped;
	image_chopped = chop_flip_image(	image_buf, 
														height, 
														width, 
														cropped,
														scaled,
														converted);

	// free image buffer
	free(image_buf);

	// return
	return image_chopped;

} 

// #ifdef __cplusplus
// }
// #endif
