#ifndef __LOGGING_H__
#define __LOGGING_H__

#include <stdio.h>

#define ml_log_error(fmt, args...) { printf("ERROR --> [%s():%d]  ",__FUNCTION__, __LINE__); printf(fmt, ## args ); fflush(stdout);}
#define ml_log_warn(fmt, args...) { printf("WARNING --> [%s():%d]  ",__FUNCTION__, __LINE__); printf(fmt, ## args ); fflush(stdout);}


#ifdef DEBUG
  #define ml_log(fmt, args...) { printf("[%s():%d]  ",__FUNCTION__, __LINE__); printf(fmt, ## args ); fflush(stdout);}
  #define ml_log_verbose(fmt, args...) { printf("[%s():%d]  ",__FUNCTION__, __LINE__); printf(fmt, ## args ); fflush(stdout);}
#else 
  #define ml_log(fmt, args...) { printf(fmt, ## args ); fflush(stdout);}
  #define ml_log_verbose(fmt, args...) {  }
#endif


#endif
