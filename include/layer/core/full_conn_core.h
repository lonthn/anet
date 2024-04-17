//
// Created by zeqi.luo on 2021/2/4.
//

#ifndef ANET_FULL_CONN_CORE_H
#define ANET_FULL_CONN_CORE_H

#include "include/math/matrix.h"

#include <mmintrin.h>
#include <xmmintrin.h>  // SSE
//#include <pmmintrin.h>  // SSE2
#include <emmintrin.h>  // SSE3

namespace anet {

static void w_mult_x_plus_b(mat_t& x, mat_t& w, mat_t& b, mat_t& y);
static void dy_mul_Tx(mat_t& dy, mat_t& x, mat_t& dw);

static void w_mult_x8_plus_b(float_t* w, float_t* x, float_t* b, float_t* z,
                              int wr, int wc, int wc_remainder) {
  __m128 m_x;
  __m128 m_w;
  __m128 y;

  int i, j;
  float_t* tx;
  float temp[4];
  for (i = 0; i < wr; ++i) {
    tx = x;
    y = _mm_setzero_ps();
    for (j = 0; j < wc; ++j) {
      /* z += x * w */
      m_x = _mm_load_ps(tx);
      m_w = _mm_load_ps(w);
      y = _mm_add_ps(y, _mm_mul_ps(m_x, m_w));
      m_x = _mm_load_ps(tx+4);
      m_w = _mm_load_ps(w+4);
      y = _mm_add_ps(y, _mm_mul_ps(m_x, m_w));
      tx+=8;
      w+=8;
    }
    for (j = 0; j < wc_remainder; ++j) {
      m_x = _mm_load_ss(tx++);
      m_w = _mm_load_ss(w++);
      y = _mm_add_ss(y, _mm_mul_ss(m_x, m_w));
    }
    _mm_store_ps(temp, y);
    /* z = xw + b */
    *(z++) = temp[0] + temp[1] + temp[2] + temp[3] + *(b++);
  }
}

static void dy_mul_Tx8(float_t* dy, float_t* x, float_t* dw,
                       int h, int w, int w_remainder) {
  __m128 a;
  __m128 m_b;

  int i, j;
  for (i = 0; i < h; ++i) {
    float_t tdy = *(dy++);
    float_t* tx_ptr = x;
    a = _mm_set_ps(tdy, tdy, tdy, tdy);
    for (j = 0; j < w; ++j) {
      m_b = _mm_load_ps(tx_ptr);
      _mm_store_ps(dw, _mm_mul_ps(a, m_b));
      m_b = _mm_load_ps(tx_ptr + 4);
      _mm_store_ps(dw+4, _mm_mul_ps(a, m_b));
      tx_ptr += 8;
      dw += 8;
    }
    for (j = 0; j < w_remainder; ++j) {
      m_b = _mm_load_ss(tx_ptr++);
      _mm_store_ss(dw++, _mm_mul_ss(a, m_b));
      //*(dw++) = tdy * *(txptr++);
    }
  }
}

static void w_mult_x_plus_b(mat_t& x, mat_t& w, mat_t& b, mat_t& y) {
  int32_t wr = w.height();
  int32_t wc = w.width();
  float_t* w_ptr = w.data();
  float_t* x_ptr = x.data();
  float_t* b_ptr = b.data();
  float_t* y_ptr = y.data();
  w_mult_x8_plus_b(w_ptr, x_ptr, b_ptr, y_ptr, wr, wc/8, wc%8);
}


void dy_mul_Tx(mat_t& dy, mat_t& x, mat_t& dw) {
  int32_t w = x.size();
  int32_t h = dy.size();
  float_t* x_ptr  = x.data();
  float_t* dw_ptr = dw.data();
  float_t* dy_ptr = dy.data();
  dy_mul_Tx8(dy_ptr, x_ptr, dw_ptr, h, w / 8, w % 8);
}

}

#endif //ANET_FULL_CONN_CORE_H
