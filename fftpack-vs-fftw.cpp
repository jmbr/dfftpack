#include <cmath>
#include <cstdlib>

#include <sys/time.h>

#include <vector>
#include <limits>
#include <complex>
#include <iomanip>
#include <iostream>
#include <valarray>

#include <fftw3.h>
#include "dfftpack.hpp"

timeval start, stop, elapsed;

inline void tic() {
    gettimeofday(&start, 0);
}

inline double toc() {
    gettimeofday(&stop, 0);
    timersub(&stop, &start, &elapsed);
    return double(elapsed.tv_sec) + 1e-6 * double(elapsed.tv_usec);
}

int main(int argc, char* argv[]) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " size repetitions" << std::endl;
    return EXIT_FAILURE;
  }

  const std::complex<double> I(0.0, 1.0);

  srand(0);

  int N = int(atof(argv[1]));
  size_t K = size_t(atof(argv[2]));

  std::vector<std::complex<double> > x1(N);
  std::vector<std::complex<double> > x2(N);

  for (int n = 0; n < N; ++n) {
    x1[n] = std::complex<double>(drand48(), drand48());
    x2[n] = x1[n];
  }

  std::vector<double> wsave(4 * N + 15);
  zffti_(&N, &wsave[0]);

  fftw_plan plan;
  fftw_complex* in  = reinterpret_cast<fftw_complex*>(&x2[0]);
  fftw_complex* out = reinterpret_cast<fftw_complex*>(&x2[0]);
  plan = fftw_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_MEASURE);

  std::valarray<double> seconds_fftpack(K), seconds_fftw(K);

  for (size_t k = 0; k < seconds_fftpack.size(); ++k) {
    tic();
    zfftf_(&N, &x1[0], &wsave[0]);
    seconds_fftpack[k] = toc();
  }

  const double avg_time_fftpack = seconds_fftpack.sum() / double(K);

  for (size_t k = 0; k < seconds_fftw.size(); ++k) {
    tic();
    fftw_execute(plan);
    seconds_fftw[k] = toc();
  }

  const double avg_time_fftw = seconds_fftw.sum() / double(K);

  std::cout << std::setprecision(8) << std::fixed
            << N << " " << avg_time_fftpack << " " << avg_time_fftw 
            << std::endl;

  fftw_destroy_plan(plan);
  return EXIT_SUCCESS;
}
