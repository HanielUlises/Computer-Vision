#include "meyer_wavelet.hpp"
#include <fftw3.h>
#include <cmath>

static constexpr double PI=3.14159265358979323846;

double Meyer2D::nu(double x){
    if(x<=0.0)return 0.0;
    if(x>=1.0)return 1.0;
    return x*x*x*(10-15*x+6*x*x);
}

double Meyer2D::psi_hat_radial(double omega){
    omega=std::abs(omega);

    if(omega<=2.0*PI/3.0)return 0.0;
    if(omega>=4.0*PI/3.0)return 0.0;

    double t=(3.0*omega/(2.0*PI))-1.0;

    if(omega<=PI)
        return std::sin(PI/2.0*nu(t));
    else
        return std::cos(PI/2.0*nu(t));
}

Meyer2D::MatrixC Meyer2D::generate(int width,int height,double scale){
    int N=width;
    int M=height;

    fftw_complex* freq=(fftw_complex*)fftw_malloc(sizeof(fftw_complex)*N*M);
    fftw_complex* spatial=(fftw_complex*)fftw_malloc(sizeof(fftw_complex)*N*M);

    double dwx=2.0*PI/N;
    double dwy=2.0*PI/M;

    for(int y=0;y<M;++y){
        for(int x=0;x<N;++x){
            int idx=y*N+x;

            double wx=(x<=N/2)?x*dwx:(x-N)*dwx;
            double wy=(y<=M/2)?y*dwy:(y-M)*dwy;

            double r=std::sqrt(wx*wx+wy*wy);
            double theta=std::atan2(wy,wx);

            double radial=psi_hat_radial(scale*r);

            std::complex<double> value=
                radial*std::exp(std::complex<double>(0.0,theta));

            freq[idx][0]=value.real();
            freq[idx][1]=value.imag();
        }
    }

    fftw_plan plan=fftw_plan_dft_2d(
        M,N,
        freq,
        spatial,
        FFTW_BACKWARD,
        FFTW_ESTIMATE
    );

    fftw_execute(plan);

    MatrixC result(M,N);

    double norm=1.0/static_cast<double>(N*M);

    for(int y=0;y<M;++y){
        for(int x=0;x<N;++x){
            int idx=y*N+x;
            result(y,x)=
                Complex(spatial[idx][0],spatial[idx][1])*norm;
        }
    }

    fftw_destroy_plan(plan);
    fftw_free(freq);
    fftw_free(spatial);

    return result;
}