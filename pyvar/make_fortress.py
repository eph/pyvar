module msvar

  use mkl95_precision, only: wp => dp
  use prior, only: loggampdf, logdirichletpdf, M_PI, determinant, logiwishpdf, lognorpdf, logigpdf
  use mkl_vsl_type
  use mkl_vsl
  use rand 

  implicit none 

  character(len=*), parameter :: mname = 'msvar_fixed_x' 
  character(len=*), parameter :: priotype = 'reduced-form'

  logical, parameter :: hyper = .true.
  integer, parameter :: use_lamxx = {use_lamxx:5.3f}
  
  integer, parameter :: p = {p}, constant = {cons}, ny = {ny}, full_T = 184
  integer, parameter :: ns_mu=1, ns_var=1, ns=ns_mu * ns_var

  integer, parameter :: nA = ny*(ny+1)/2, nF = ny**2*p + ny*constant, nQ = ns_mu*(ns_mu-1) + ns_var*(ns_var-1)
  integer, parameter :: nlam = 4
  integer, parameter :: npara = ns_mu*(nA + nF) + (ns_var-1)*ny + nQ
  integer, parameter :: T = full_T - p


  character(len=*), parameter :: datafile = '/mq/home/m1eph00/projects/var-smc/fortran/data/sz_2008_joe_data_short.csv'
  character(len=*), parameter :: AFmufile = '/mq/home/m1eph00/projects/var-smc/fortran/prior/AFmu_final.txt'
  character(len=*), parameter :: AFvarfile = '/mq/home/m1eph00/projects/var-smc/fortran/prior/AFsigma_final.txt'
  character(len=*), parameter :: Xiprifile = '/mq/home/m1eph00/projects/var-smc/fortran/prior/Xiprior.txt'
  character(len=*), parameter :: Qmuprifile = '/mq/home/m1eph00/projects/var-smc/fortran/prior/Qmualpha.txt'
  character(len=*), parameter :: Qvarprifile = '/mq/home/m1eph00/projects/var-smc/fortran/prior/Qvaralpha.txt'

  real(wp) :: priAF_mu(ns_mu*(nA+nF)), priAF_var(ns_mu*(nA+nF),nA+nF)
  real(wp) :: priXi_mean((ns_var-1)*ny), priXi_std((ns_var-1)*ny)
  real(wp) :: priQmu(ns_mu,ns_mu), priQvar(ns_var,ns_var)

  ! for reduced form prior -- IWS is unfortunately hard coded right now!
  real(wp) :: M(9,6),Dnplus(6,9),Nn(9,9)

  !  integer :: pshape(npara), pmask(npara)
  real(wp) :: trspec(4,npara)
  integer :: pshape(npara), pmask(npara)
  real(wp) :: pmean(npara), pstdd(npara), pfix(npara)

  character(len=*), parameter :: transfile = ''
  character(len=*), parameter :: initfile= ''
  character(len=*), parameter :: initwt =  ''
  character(len=*), parameter :: priorfile =  ''
  integer, parameter :: nobs = T

  real(wp) :: data(full_T,ny)
  logical :: data_loaded
  real(wp) :: YY(T,ny), XX(T,ny*p+constant)


  real(wp), parameter :: hyper_lam1 = {lam1:f}_wp, hyper_lam2 = {lam2:f}_wp, hyper_lam3 = {lam3:f}_wp
  real(wp), parameter :: hyper_lam4 = {lam4:f}_wp, hyper_lam5 = {lam5:f}_wp, hyper_lamxx = {lamxx:f}_wp
  real(wp), parameter :: hyper_tau = {tau:f}_wp

  real(wp), parameter, dimension(ny) :: hyper_ybar = {ybar}
  real(wp), parameter, dimension(ny) :: hyper_sbar = {sbar}

  real(wp) :: hyper_phistar(nF), hyper_Omega_inv(nF/ny,nF/ny), hyper_iw_Psi(ny,ny)
  integer :: hyper_iw_nu 

contains
  include '/mq/home/m1eph00/projects/var-smc/fortran/libmsvar.f90'

  subroutine load_data()

    integer :: i,j

    open(1, file=datafile, status='old', action='read')
    do i = 1, full_T
       read(1, *) data(i,:)
    end do
    close(1)    
    XX = 1.0_wp

    do i = 1,T

       YY(i,:) = data(i+p,:)

       do j = 1,p
          XX(i,(j-1)*ny+1:j*ny) = data(i+p-j,:)

       end do
       !write(*,'(100f5.2)') XX(i,:)
    end do
    data_loaded = .true.

    !------------------------------------------------------------
    ! prior
    !------------------------------------------------------------
    if (priotype=='reduced-form') then 
       call construct_prior_hyper(hyper_lam1, hyper_lam2, hyper_lam3, hyper_lam4, hyper_lam5, &
            hyper_tau, hyper_lamxx, hyper_ybar, hyper_sbar, &
            hyper_phistar, hyper_Omega_inv, hyper_iw_Psi, hyper_iw_nu)
       ! hyper_iw_nu = 10
       ! load the matrix derivatives 
       open(1,file='./prior/Dnplus.txt',status='old',action='read')
       open(2,file='./prior/Nn.txt',status='old',action='read')
       open(3,file='./prior/M.txt',status='old',action='read')
       do i = 1,6
          read(1,*) Dnplus(i,:)
       end do
       do i = 1,9
          read(2,*) Nn(i,:)
          read(3,*) M(i,:)
       end do
       close(1)
       close(2)
       close(3)
    else
       open(1,file=AFmufile, status='old', action='read')
       open(2,file=AFvarfile, status='old', action='read')
       do i = 1,ns_mu*(nA + nF)
          read(1,*) priAF_mu(i)
          read(2,*) priAF_var(i,:)
       end do
       close(1)
       close(2)
    end if

    if (ns_var > 1) then
       open(1,file=Xiprifile,status='old',action='read')
       do i = 1,(ns_var-1)*ny
          read(1,*) priXi_mean(i), priXi_std(i)
       end do
       close(1)

       open(1,file=Qvarprifile,status='old',action='read')
       do i = 1,ns_var
          read(1,*) priQvar(:,i)
       end do
       close(1)
    end if


    if (ns_mu > 1) then
       open(1,file=Qmuprifile,status='old',action='read')
       do i = 1,ns_mu
          read(1,*) priQmu(:,i)
       end do
       close(1)
    end if

    pmask = 0
  end subroutine load_data


  function pr(para) result(logprior)

    real(wp), intent(in) :: para(npara)
    real(wp) :: logprior

    real(wp) :: b, a
    integer :: ind0, ind1,i

    ! for reduced-form prior 
    real(wp) :: A0(ny,ny), SIGMA(ny,ny), negSIGMAkrSIGMA(ny**2,ny**2), A0krI(ny**2,ny**2), eye(ny,ny)
    integer :: j, As_ind0,info, ipiv(ny), col
    real(wp) :: work(3*ny), test
    real(wp) :: temp1(nA,ny**2),temp2(nA,ny**2),temp3(nA,nA), det1,det2, SIGMAcp(ny,ny), temp4(nA,nA)
    real(wp) :: mvn_covar(nF,nF), PHIvec(nF), F(nF/ny,ny), PHI(nF/ny,ny), A0i(ny,ny)


    priAF_mu = 0.0_wp

    ! AF
    logprior = 0.0_wp
    do i = 1, ns_mu
       ind0 = (i-1)*(nA+nF)+1
       ind1 = i*(nA+nF)
       if (not(priotype=='reduced-form')) then 
          logprior = logprior + mvnormal_pdf(para(ind0:ind1), priAF_mu(ind0:ind1), priAF_var(ind0:ind1,:))
       else
          !------------------------------------------------------------
          ! MVN-IW
          !------------------------------------------------------------
          A0 = 0.0_wp
          As_ind0 = ind0-1
          eye = 0.0_wp
          ! unroll A0
          do j = 1,ny
             A0(1:j,j) = para(As_ind0+1:As_ind0+j)
             As_ind0 = As_ind0 + j
             eye(j,j) = 1.0_wp
          end do


          ! unroll Fvec
          A0i = A0
          call dgetrf(ny,ny,A0i,ny,ipiv,info)
          call dgetri(ny,A0i,ny,ipiv,work,3*ny,info)

          col = nF/ny
          do j = 1,ny
             !F((j-1)*col+1:j*col) = para((i-1)*nF + i*nA + (j-1)*col+1:(i-1)*nF+i*nA+j*col)
             F(:,j) = para((i-1)*nF + i*nA + (j-1)*col+1:(i-1)*nF+i*nA+j*col)
          end do


          call dgemm('n','n',col,ny,ny,1.0_wp,F,col,A0i,ny,0.0_wp,PHI,col)
          do j = 1,ny
             PHIvec((j-1)*col+1:j*col) = PHI(:,j)
          end do

          
          call dgemm('n','t',ny,ny,ny,1.0_wp,A0,ny,A0,ny,0.0_wp,SIGMA,ny)
          call dgetrf(ny,ny,SIGMA,ny,ipiv,info)
          call dgetri(ny,SIGMA,ny,ipiv,work,3*ny,info)

          call Kronecker(ny,ny,nF/ny,nF/ny,nF,nF,0.0_wp,1.0_wp,SIGMA,hyper_Omega_inv,mvn_covar)

          SIGMAcp = SIGMA
          test = 0.0_wp
          test = test + logiwishpdf(hyper_iw_nu,hyper_iw_Psi(1:ny,1:ny),SIGMAcp,ny)

          call Kronecker(ny,ny,ny,ny,ny**2,ny**2,0.0_wp,1.0_wp,-SIGMA,SIGMA,negSIGMAkrSIGMA)
          call Kronecker(ny,ny,ny,ny,ny**2,ny**2,0.0_wp,1.0_wp, A0, eye, A0krI)
          call dgemm('n','n',nA,ny**2,ny**2,1.0_wp,Dnplus,nA,negSIGMAkrSIGMA,ny**2,0.0_wp,temp1,nA)
          call dgemm('n','n',nA,ny**2,ny**2,2.0_wp,temp1,nA,Nn,ny**2,0.0_wp,temp2,nA)

          call dgemm('n','n',nA,ny**2,ny**2,1.0_wp,temp2,nA,A0krI,ny**2,0.0_wp,temp1,nA) 
          call dgemm('n','n',nA,nA,ny**2,1.0_wp,temp1,nA,M,ny**2,0.0_wp,temp3,nA)
          call dgemm('n','t',nA,nA,nA,1.0_wp,temp3,nA,temp3,nA,0.0_wp,temp4,nA)

          det1 = determinant_gen(temp4,nA)
          
          call dgemm('n','t',ny,ny,ny,1.0_wp,A0,ny,A0,ny,0.0_wp,SIGMAcp,ny)

          ! testing only 
          test = test + log(abs(det1))/2.0_wp -1.0_wp*(ny*p+1)*log(abs(determinant_gen(A0,ny)))!
  
          call dpotrf('l',nF,mvn_covar,nF,info)
          test = test + mvnormal_pdf(PHIvec, hyper_phistar, mvn_covar)

       end if
       logprior = logprior + test

    end do
    
    
    ! Xi 
    ind0 = ns_mu*(nA+nF)
    do i = 1,ny*(ns_var-1)
       b = priXi_std(i)**2/priXi_mean(i); !theta
       a = priXi_mean(i)/b;               !k
       a = 1.0_wp
       b = 1.0_wp

       if (para(ind0+i) < 0) then
          logprior = -100000000000000.0_wp
          return
       end if

       !print*,loggampdf(para(ind0+i),a,b), para(ind0+i)
       logprior = logprior + loggampdf(para(ind0+i),a,b)

    end do


    ind0 = ns_mu*(nA+nF) + (ns_var-1)*ny
    if (ns_mu > 1) then
       do i = 1,ns_mu
          logprior = logprior + logdirichletpdf(para(ind0+1:ind0+(ns_mu-1)),priQmu(:,i),ns_mu)
          ind0=ind0+ns_mu-1
       end do
    end if

    ind0 = ns_mu*(nA+nF) + (ns_var-1)*ny + ns_mu*(ns_mu-1)
    if (ns_var > 1) then
       do i = 1,ns_var
          logprior = logprior + logdirichletpdf(para(ind0+1:ind0+(ns_var-1)),priQvar(:,i),ns_var)
          ind0 = ind0+ns_var-1
       end do
    end if

  end function pr

  function priorrand(npart, pshape, pmean, pstdd, pmask, pfix) result(priodraws)

    integer, intent(in) :: npart

    real(wp), intent(in) :: pmean(npara),pstdd(npara),pfix(npara)
    integer, intent(in) :: pshape(npara),pmask(npara)
    real(wp) :: priodraws(npart, npara)
    real(wp) :: mark_priodraws(npart, npara+ny)

    real(wp) :: mu(nA+nF),Lt(nA+nF,nA+nF), dev(npart,nA+nF)
    integer :: i,z,ind0,j,k,jj

    real(wp) :: dir_mu(npart, ns_mu), dir_var(npart,ns_var)
    type (VSL_STREAM_STATE) :: stream
    integer(kind=4) errcode
    integer brng, method, methodu, methodg, methodb, seed, mstorage, time_array(8), status

    real(wp):: a,b

    ! for reduced form prior
    real(wp) :: chol_S(ny,ny), iW(ny,ny), mvn_covar(nF,nF), dev_F(nF)
    real(wp) :: SIGMAi(ny,ny), Fvec(nF), F(nF/ny,ny), Aplus(nF/ny,ny)
    integer :: info, ipiv(ny)
    real(wp) :: work(3*ny)

    real(wp) :: dev2(npart,ns_mu*(nA+nF))

    real(wp) :: lam1, lam2, lam3, lam4, lam5, tau, lamxx, ybar(ny), sbar(ny), lam1_rvs(2)
    real(wp), allocatable :: dev_iw(:,:), W(:,:), rand_temp(:)


    brng=VSL_BRNG_MT19937
    call date_and_time(values=time_array)
    seed=mod(sum(time_array),10000)

    errcode = vslnewstream(stream, brng, seed)
    methodg=VSL_METHOD_DGAMMA_GNORM

    ! AF
    do i = 1, ns_mu
       if (not(priotype=='reduced-form')) then
          mu = 0.0_wp !priAF_mu((i-1)*(nA+nF)+1:i*(nA+nF))
          z = nA+nF
          Lt = transpose(priAF_var((i-1)*(nA+nF)+1:i*(nA+nF),:))
          ! status = vdrnggaussianmv(VSL_RNG_METHOD_GAUSSIANMV_BOXMULLER2, stream, & 
          !      npart, dev, z, VSL_MATRIX_STORAGE_FULL, mu, Lt)
          status = vdrnggaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER, stream, & 
               npart*(nA+nF), dev,0.0_wp,1.0_wp)
          call dgemm('n','t',npart,nA+nF,nA+nF,1.0_wp,dev,npart,Lt,nA+nF,0.0_wp,dev2,npart)
          priodraws(:,(i-1)*(nA+nF)+1:i*(nA+nF)) = dev2
       else 

          do j = 1,npart

             chol_S = hyper_iw_Psi(1:ny,1:ny)
             ! C = chol(inv(S),'lower')
             call dgetrf(ny,ny,chol_S,ny,ipiv,info)
             call dgetri(ny,chol_S,ny,ipiv,work,3*ny,info)
             call dpotrf('l',ny,chol_S,ny,info)
             do jj = 1,ny-1
                chol_S(jj+1:ny,jj) = 0.0_wp
             end do

             allocate(dev_iw(ny,hyper_iw_nu), W(ny,hyper_iw_nu))

             status = vdrnggaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER, stream, & 
                  ny*hyper_iw_nu, dev_iw, 0.0_wp, 1.0_wp)
             call dgemm('n','n',ny,hyper_iw_nu,ny,1.0_wp,chol_S,ny,dev_iw,ny,0.0_wp,W,ny)
             call dgemm('n','t',ny,ny,hyper_iw_nu,1.0_wp,W,ny,W,ny,0.0_wp,iW,ny)

             call dgetrf(ny,ny,iW,ny,ipiv,info)
             call dgetri(ny,iW,ny,ipiv,work,3*ny,info)


             ! ! store relevant parts of iW
             call Kronecker(ny,ny,nF/ny,nF/ny,nF,nF,0.0_wp,1.0_wp,iW,hyper_Omega_inv,mvn_covar)
             call dpotrf('l',nF,mvn_covar,nF,info)
             do k = 1,nF-1
                mvn_covar(k,k+1:nF) = 0.0_wp
             end do

             status = vdrnggaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER, stream, & 
                  nF, dev_F, 0.0_wp, 1.0_wp)       
             Fvec = hyper_phistar

             call dgemv('n',nF,nF,1.0_wp,mvn_covar,nF,dev_F,1,1.0_wp,Fvec,1)
             do k = 1,ny
                F(:,k) = Fvec((k-1)*nF/ny+1:k*nF/ny)
             end do

             !------------------------------------------------------------
             ! SAVE A0 
             SIGMAi = iW
             call dpotrf('u',ny,SIGMAi,ny,info)
             do k = 1,ny-1
                SIGMAi(k+1:ny,k) = 0.0_wp
             end do
             call dgetrf(ny,ny,SIGMAi,ny,ipiv,info)
             call dgetri(ny,SIGMAi,ny,ipiv,work,3*ny,info)

             ind0 = 1
             do k = 1,ny
                priodraws(j,(i-1)*(nA+nF)+ind0:(i-1)*(nA+nF)+ind0+k-1) = SIGMAi(1:k,k)

                ind0 = ind0 + k

                ! zero out for later construction of Aplus
                ! if (k < ny) then
                !    SIGMAi(k+1:ny,k) = 0.0_wp
                ! end if
             end do
             !------------------------------------------------------------

             call dgemm('n','n',nF/ny,ny,ny,1.0_wp,F,nF/ny,SIGMAi,ny,0.0_wp,Aplus,nF/ny)
             do k = 1,ny
                priodraws(j,(i-1)*(nA+nF)+nA+(k-1)*nF/ny+1:(i-1)*(nA+nF)+nA+k*nF/ny) = Aplus(:,k)
             end do

             deallocate(dev_iw, W)

          end do

       end if

    end do

    ind0 = ns_mu*(nA+nF)
    do i = 1,ny*(ns_var-1)
       b = priXi_std(i)**2/priXi_mean(i); !theta
       a = priXi_mean(i)/b;               !k
       a = 1.0_wp
       b = 1.0_wp
       status = vdrnggamma(methodg, stream, npart, priodraws(:, ind0+i), a, 0.0_wp, b)
    end do

    ind0 = ns_mu*(nA+nF) + (ns_var-1)*ny

    if (ns_mu > 1) then
       do i = 1,ns_mu
          do j = 1,ns_mu
             status = vdrnggamma(methodg, stream, npart, dir_mu(:, j), priQmu(j,i), 0.0_wp, 1.0_wp)
          end do
          do j = 1,npart
             dir_mu(j,:) = dir_mu(j,:)/sum(dir_mu(j,:))
          end do
          priodraws(:,ind0+1:ind0+ns_mu-1) = dir_mu(:,1:ns_mu-1)

          ind0 = ind0 + ns_mu - 1
       end do
    end if


    ind0 = ns_mu*(nA+nF) + (ns_var-1)*ny + ns_mu*(ns_mu-1)

    if (ns_var > 1) then
       do i = 1,ns_var
          do j = 1,ns_var
             status = vdrnggamma(methodg, stream, npart, dir_var(:, j), priQvar(j,i), 0.0_wp, 1.0_wp)
          end do
          do j = 1,npart
             dir_var(j,:) = dir_var(j,:)/sum(dir_var(j,:))
          end do
          priodraws(:,ind0+1:ind0+ns_var-1) = dir_var(:,1:ns_var-1)
          ind0 = ind0 + ns_var - 1
       end do
    end if

  end function priorrand



end module msvar
