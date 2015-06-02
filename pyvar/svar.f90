module {name}

  use mkl95_precision, only: wp => dp
  use prior, only: loggampdf, logdirichletpdf, M_PI, determinant, lognorpdf
  use mkl_vsl_type
  use mkl_vsl
  use rand 

  implicit none 

  character(len=*), parameter :: mname = '{name}'

  integer, parameter :: p = {p}, constant = {cons:d}, ny = {ny}, full_T = {T}
  integer, parameter :: nA = {nA}, nF = {nF}, npara = {npara}
  integer, parameter :: T = full_T - p

  character(len=*), parameter :: datafile = '{datafile}'
  character(len=*), parameter :: AFmufile = '{AFmufile}'
  character(len=*), parameter :: AFvarfile = '{AFvarfile}'

  real(wp) :: trspec(4,npara)
  integer :: pshape(npara), pmask(npara)
  real(wp) :: pmean(npara), pstdd(npara), pfix(npara)
  character(len=*), parameter :: transfile = ''
  character(len=*), parameter :: initfile= ''
  character(len=*), parameter :: initwt =  ''
  character(len=*), parameter :: priorfile =  ''
  integer, parameter :: nobs = T

  real(wp) :: priAF_mu((nA+nF)), priAF_var((nA+nF),nA+nF)

  real(wp) :: data(full_T,ny)
  logical :: data_loaded
  real(wp) :: YY(T,ny), XX(T,ny*p+constant)

  real(wp), parameter :: pmeanrho = 0.75d0, pstddrho = 0.10d0
  real(wp), parameter :: pmeanbeta = 0.0d0, pstddbeta = 0.10d0

contains

  function lik(para) result(lk)
   
    real(wp), intent(in):: para(npara)
    real(wp) :: lk
   
    real(wp) :: A0(ny,ny), F(ny*p+constant,ny), A0A0p(ny,ny)
    integer :: RC
   
    real(wp) :: resid(T,ny)

    if (data_loaded == .false.) then
       call load_data()
    end if
   
    A0 = 0.0d0
    F = 0.0d0      
{assign_para}
    lk = -T*ny/2.0d0*log(2.0_wp*M_PI)
   
    call dgemm('n','n',T,ny,ny,1.0_wp,YY,T,A0,ny,0.0_wp,resid,T)
    call dgemm('n','n',T,ny,ny*p+constant,-1.0_wp,XX,T,F,ny*p+constant,1.0_wp,resid,T)
    
    call dgemm('n','t',ny,ny,ny,1.0_wp,A0,ny,A0,ny,0.0_wp,A0A0p,ny)
   
    lk = lk + -0.5d0 * sum(resid**2) + T*0.5d0*log(determinant(A0A0p, ny))
   
  end function lik

    function pr(para) result(logprior)

    real(wp), intent(in) :: para(npara)
    real(wp) :: logprior

    ! AF
    logprior = mvnormal_pdf(para(:nA+nF), priAF_mu, priAF_var)

    !------------------------------------------------------------
    ! hack for external instruments model
    !------------------------------------------------------------
    if (npara>(nA+nF)) then
       logprior = logprior + lognorpdf(para(nA+nF+1),0.0d0,pstddbeta)

       if ( (para(nA+nF+2)<0.0d0) .or. (para(nA+nF+2)>1.0d0) ) then
          logprior = -1000000000.0d0
       end if
    end if

  end function pr


  function priorrand(npart, pshape, pmean, pstdd, pmask, pfix,seed0,stream0) result(priodraws)

    integer, intent(in) :: npart

    real(wp), intent(in) :: pmean(npara),pstdd(npara),pfix(npara)
    integer, intent(in) :: pshape(npara),pmask(npara)
    real(wp) :: priodraws(npart, npara)

    integer, optional :: seed0
    type(VSL_STREAM_STATE), optional :: stream0

    real(wp) :: mu(nA+nF),Lt(nA+nF,nA+nF), dev2(npart,(nA+nF)), dev(npart,(nA+nF))
    integer :: i,z,ind0,j,k

    type (VSL_STREAM_STATE) :: stream
    integer(kind=4) errcode
    integer brng, method, methodu, methodg, methodb, seed, mstorage, time_array(8), status

    real(wp):: a,b

    brng=VSL_BRNG_MT19937
    call date_and_time(values=time_array)
    seed=mod(sum(time_array),10000)

    errcode = vslnewstream(stream, brng, seed)
    methodg=VSL_METHOD_DGAMMA_GNORM

    Lt = transpose(priAF_var)

    status = vdrnggaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER, stream, & 
         npart*(nA+nF), dev,0.0_wp,1.0_wp)

    call dgemm('n','t',npart,nA+nF,nA+nF,1.0_wp,dev,npart,Lt,nA+nF,0.0_wp,dev2,npart)
    priodraws(:,1:(nA+nF)) = dev2

    !------------------------------------------------------------
    ! hack for external instruments model
    !------------------------------------------------------------
    if (npara > (nA+nF)) then
       status = vdrnggaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER, stream, & 
            npart*(nA+nF), dev,0.0_wp,1.0_wp)
       priodraws(:,nA+nF+1) = pstddbeta*dev(:,1)
       status = vdrnguniform(VSL_METHOD_DUNIFORM_STD, stream, npart, priodraws(:,nA+nF+2), 0.0d0, 1.0d0)
    end if

  end function priorrand

  subroutine write_model_para(fstr)
    character(len=*), intent(in) :: fstr

  end subroutine write_model_para

  function likT(para,ti) result(l)
    integer, intent(in) :: ti
    real(wp), intent(in) :: para(npara)
    real(wp) :: l

  end function likT
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

    end do
    data_loaded = .true.

    ! load AFmu, AFSigm
    open(1,file=AFmufile, status='old', action='read')
    open(2,file=AFvarfile, status='old', action='read')
    do i = 1,nA+nF
       read(1,*) priAF_mu(i)
       read(2,*) priAF_var(i,:)
    end do
    close(1)
    close(2)

  end subroutine load_data

  subroutine print_coeff(para)

    real(wp) :: para(npara)
    real(wp) :: A0(ny,ny), F(ny*p+constant,ny)

    integer :: i

    A0 = 0.0d0
    F = 0.0d0
{assign_para}
    print *,'A0 = '
    do i = 1,ny
       write(*,'(100f5.3)') A0(i,:)
    end do

    print *, 'F = '
    do i = 1,ny*p+constant
       write(*,'(100f5.3)') F(i,:)
    end do
  end subroutine print_coeff


end module {name}
