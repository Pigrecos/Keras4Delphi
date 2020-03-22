{*******************************************************}
{                                                       }
{       Numpy Wrapper for Delphi                        }
{                                                       }
{       Copyright (C) 2020 Pigrecos(Max)                }
{                                                       }
{*******************************************************}
unit np.Api;

//np.array_manipulation.gen.cs
//np.math.gen.cs
//np.array_creation.gen.cs
//np.array.cs
//np.random.cs

interface
   uses  System.Rtti,
         PythonEngine,
         Python.Utils,

         np.Models,
         np.Utils,
         np.Base;

type

  TNumPyArray = class Helper for  TNumPy
    public
      //np.sorting.gen.cs
      function argmax(a: TNDarray ; axis: PInteger ; var _out : TNDarray): TNDarray;
      //np.staticstics.gen.cs
      function amin(a: TNDarray ; axis: TArray<Integer> ; var _out : TNDarray; keepdims : PBoolean; initial : TValue): TNDarray;
      function amax(a: TNDarray ; axis: TArray<Integer> ; var _out : TNDarray; keepdims : PBoolean; initial : TValue): TNDarray;
      function min(a: TNDarray ; axis: TArray<Integer> ; var _out : TNDarray; keepdims : PBoolean; initial : TValue): TNDarray; // np.aliases.cs
      function max(a: TNDarray ; axis: TArray<Integer> ; var _out : TNDarray; keepdims : PBoolean; initial : TValue): TNDarray; // np.aliases.cs

      //np.array_manipulation.gen.cs
      class function reshape(a: TNDarray; newshape: Tnp_Shape; order : PChar = nil): TNDarray;overload;
      class function reshape(a: TNDarray; newshape: TArray<Integer>): TNDarray;overload;   // np.aliases.cs

      function &repeat(a : TNDarray  ; repeats : TArray<Integer>  ; axis : PInteger = nil): TNDarray;
      function append(arr : TNDarray  ; values : TNDarray  ; axis : PInteger = nil): TNDarray;
      function asarray_chkfinite(a : TNDarray  ; dtype : TDtype = nil; order : PChar = nil): TNDarray;
      function asfarray(a : TNDarray  ; dtype : TDtype = nil): TNDarray;
      function asfortranarray(a : TNDarray  ; dtype : TDtype = nil): TNDarray;
      function atleast_1d(arys: TArray<TNDarray>): TNDarray;
      function atleast_2d(arys: TArray<TNDarray>): TNDarray;
      function atleast_3d(arys: TArray<TNDarray>): TNDarray;
      function broadcast(in2, in1: TNDarray): TNDarray;
      function broadcast_arrays(args : TArray<TNDarray>  ; subok : PBoolean = nil): TArray<TNDarray>;
      function broadcast_to(arr : TNDarray  ; shape : Tnp_Shape  ; subok : Boolean = false): TNDarray;
      function column_stack(tup: TArray<TNDarray>): TNDarray;
      function concatenate(arys: TArray<TNDarray>; axis: Integer; var _out: TNDarray): TNDarray;
      procedure copyto(dst : TNDarray  ; src : TNDarray  ; casting : string = 'same_kind'; where : TNDarray = nil); overload;
      Procedure copyto(dst : TNDarray  ; src : TNDarray  ; casting : string = 'same_kind'; where : TArray<Boolean> = []); overload;
      function delete(arr : TNDarray  ; obj : Tnp_Slice  ; axis : PInteger = nil): TNDarray;
      function dstack(tup: TArray<TNDarray>): TNDarray;
      function expand_dims(a: TNDarray; axis: Integer): TNDarray;
      function flatten(order : PChar = nil): TNDarray;
      function flip(m : TNDarray  ; axis : TArray<Integer> = []): TNDarray;
      function fliplr(m: TNDarray): TNDarray;
      function flipud(m: TNDarray): TNDarray;
      function hstack(tup: TArray<TNDarray>): TNDarray;
      function insert(arr : TNDarray  ; obj : Integer = 0; values : TNDarray  = nil; axis : PInteger = nil): TNDarray;
      function moveaxis(a: TNDarray; source, destination: TArray<Integer>): TNDarray;
      function ravel(a : TNDarray  ; order : PChar = nil): TNDarray;
      function require(a : TNDarray  ; dtype : TDtype  ; requirements : TArray<string>= []): TNDarray;
      function roll(a : TNDarray  ; shift : TArray<Integer>  ; axis : TArray<Integer> = []): TNDarray;
      function rollaxis(a : TNDarray  ; axis : Integer ; start : Integer = 0): TNDarray;
      function rot90(m : TNDarray  ; k : Integer = 1; axes : TArray<Integer> = []): TNDarray;
      function split(ary : TNDarray  ; indices_or_sections : TArray<Integer>; axis : Integer = 0): TArray<TNDarray>;
      function squeeze(a : TNDarray  ; axis : TArray<Integer> = []): TNDarray;
      function stack(arrays: TArray<TNDarray>; axis: Integer; var _out: TNDarray): TNDarray;
      function swapaxes(a: TNDarray; axis1, axis2: Integer): TNDarray;
      function tile(A, reps: TNDarray): TNDarray;
      function transpose(a : TNDarray  ; axes : TArray<Integer> = []): TNDarray;
      function trim_zeros(filt : TNDarray  ; trim : string = 'fb'): TNDarray;

      function unique(ar : TNDarray  ; axis : PInteger = nil): TNDarray; overload;
      Function unique(ar : TNDarray  ; return_index: Boolean= False ; return_inverse: Boolean=False; return_counts: Boolean=false; axis : PInteger = nil):TArray<TNDarray>; overload;

      Function vstack(tup: TArray<TNDarray>): TNDarray;
      //
      //np.math.gen.cs
      function maximum(x2, x1: TNDarray; var _out: TNDarray; var where : TNDarray): TNDarray; overload;
      function maximum(x2, x1: TNDarray; var _out: TNDarray): TNDarray; overload;
      function maximum(x2, x1: TNDarray ): TNDarray; overload;
      function square(x: TNDarray; var _out, where: TNDarray): TNDarray; overload;
      function square(x: TNDarray; var _out:TNDarray): TNDarray; overload;
      function square(x: TNDarray): TNDarray;  overload;
      function &mod(x1, x2: TNDarray; var _out, where: TNDarray): TNDarray;
      function &absolute(x: TNDarray; var _out, where: TNDarray): TNDarray;
      function abs(x: TNDarray; var _out, where: TNDarray): TNDarray;  //np.aliases.cs
      function add(x2, x1: TNDarray; var _out, where: TNDarray): TNDarray;
      function angle(z: TNDarray ; deg : Boolean = false): TNDarray;
      function arccos(x: TNDarray; var _out, where: TNDarray): TNDarray;
      function arccosh(x: TNDarray; var _out, where: TNDarray): TNDarray;
      function arcsin(x: TNDarray; var _out, where: TNDarray): TNDarray;
      function arcsinh(x: TNDarray; var _out, where: TNDarray): TNDarray;
      function arctan(x: TNDarray; var _out, where: TNDarray): TNDarray;
      function arctan2(x1, x2: TNDarray; var _out, where: TNDarray): TNDarray;
      function arctanh(x: TNDarray; var _out, where: TNDarray): TNDarray;
      function around(a: TNDarray; decimals: Integer; var _out: TNDarray): TNDarray;
      function cbrt(x: TNDarray; var _out, where: TNDarray): TNDarray;
      function ceil(x: TNDarray; var _out, where: TNDarray): TNDarray;
      function clip(a, a_min, a_max: TNDarray; var _out: TNDarray): TNDarray;
      function conj(x: TNDarray; var _out, where: TNDarray): TNDarray;
      function convolve(a: TNDarray ; v: TNDarray ; mode: String = 'full'): TNDarray;
      function copysign(x1, x2: TNDarray; var _out, where: TNDarray): TNDarray;
      function cos(x: TNDarray; var _out, where: TNDarray): TNDarray;
      function cosh(x: TNDarray; var _out, where: TNDarray): TNDarray;
      function cross(a: TNDarray ; b: TNDarray ; axisa: Integer = -1; axisb: INteger = -1; axisc: Integer = -1; axis: PInteger = nil): TNDarray;
      function cumprod(a: TNDarray; axis: PInteger; dtype: TDtype; var _out: TNDarray): TNDarray;
      function cumsum(a: TNDarray; axis: PInteger; dtype: TDtype; var _out: TNDarray): TNDarray;
      function deg2rad(x: TNDarray; var _out, where: TNDarray): TNDarray;
      function degrees(x: TNDarray; var _out, where: TNDarray): TNDarray;
      function diff(a: TNDarray ; n: Integer = 1; axis: INteger = -1; append: TNDarray = nil; prepend: TNDarray = nil): TNDarray;
      function divide(x1, x2: TNDarray; var _out, where: TNDarray): TNDarray;
      function divmod(x1, x2: TNDarray; var _out, where: TNDarray): TArray<TNDarray>;
      function ediff1d(ary: TNDarray ; to_end: TNDarray = nil; to_begin: TNDarray = nil): TNDarray;
      function exp(x: TNDarray; var _out, where: TNDarray): TNDarray;
      function exp2(x: TNDarray; var _out, where: TNDarray): TNDarray;
      function expm1(x: TNDarray; var _out, where: TNDarray): TNDarray;
      function fabs(x: TNDarray; var _out, where: TNDarray): TNDarray;
      function fix(x: TNDarray ; y: TNDarray = nil): TNDarray;
      function float_power(x1, x2: TNDarray; var _out, where: TNDarray): TNDarray;
      function floor(x: TNDarray; var _out, where: TNDarray): TNDarray;
      function floor_divide(x1, x2: TNDarray; var _out, where: TNDarray): TNDarray;
      function fmax(x2, x1: TNDarray; var _out, where: TNDarray): TNDarray;
      function fmin(x2, x1: TNDarray; var _out, where: TNDarray): TNDarray;
      function fmod(x1, x2: TNDarray; var _out, where: TNDarray): TNDarray;
      function frexp(x, out1, out2: TNDarray; var _out, where: TNDarray): TArray<TNDarray>;
      function gcd(x2, x1: TNDarray): TNDarray;
      function gradient(f: TNDarray ; vararg : TNDarray = nil; edge_order : PInteger = nil; axis: TArray<Integer> = []): TNDarray;
      function heaviside(x1, x2: TNDarray; var _out, where: TNDarray): TNDarray;
      function hypot(x2, x1: TNDarray; var _out, where: TNDarray): TNDarray;
      function imag(val: TNDarray): TNDarray;
      function lcm(x2, x1: TNDarray): TNDarray;
      function ldexp(x1, x2: TNDarray; var _out, where: TNDarray): TNDarray;
      function log(x: TNDarray; var _out, where: TNDarray): TNDarray;
      function log10(x: TNDarray; var _out, where: TNDarray): TNDarray;
      function log1p(x: TNDarray; var _out, where: TNDarray): TNDarray;
      function log2(x: TNDarray; var _out, where: TNDarray): TNDarray;
      function logaddexp(x2, x1: TNDarray; var _out, where: TNDarray): TNDarray;
      function logaddexp2(x2, x1: TNDarray; var _out, where: TNDarray): TNDarray;
      function minimum(x2, x1: TNDarray; var _out, where: TNDarray): TNDarray;
      function modf(x: TNDarray; var _out, where: TNDarray): TArray<TNDarray>;
      function multiply(x2, x1: TNDarray; var _out, where: TNDarray): TNDarray;
      function nan_to_num(x: TNDarray ; copy: Boolean = true): TNDarray;
      function nancumprod(a: TNDarray; axis: PInteger; dtype: TDtype; var _out: TNDarray): TNDarray;
      function nancumsum(a: TNDarray; axis: PInteger; dtype: TDtype; var _out: TNDarray): TNDarray;
      function nanprod(a: TNDarray ; axis: TArray<Integer> ; dtype: TDtype  ; var _out : TNDarray; keepdims : PBoolean = nil): TNDarray;
      function nansum(a: TNDarray ; axis: TArray<Integer> ; dtype: TDtype  ; var _out : TNDarray; keepdims : PBoolean = nil): TNDarray;
      function negative(x: TNDarray; var _out, where: TNDarray): TNDarray;
      function nextafter(x1, x2: TNDarray; var _out, where: TNDarray): TNDarray;
      function positive(x: TNDarray): TNDarray;
      function power(x1, x2: TNDarray; var _out, where: TNDarray): TNDarray;
      function prod(a: TNDarray; axis: TArray<Integer>; dtype: TDtype; var _out: TNDarray; keepdims: PBoolean; initial: TValue): TNDarray;
      function rad2deg(x: TNDarray; var _out, where: TNDarray): TNDarray;
      function radians(x: TNDarray; var _out, where: TNDarray): TNDarray;
      function real(val: TNDarray): TNDarray;
      function real_if_close(a: TNDarray ; tol: double = 100): TNDarray;
      function reciprocal(x: TNDarray; var _out, where: TNDarray): TNDarray;
      function remainder(x1, x2: TNDarray; var _out, where: TNDarray): TNDarray;
      function rint(x: TNDarray; var _out, where: TNDarray): TNDarray;
      function sign(x: TNDarray; var _out, where: TNDarray): TNDarray;
      function signbit(x: TNDarray; var _out, where: TNDarray): TNDarray;
      function sin(x: TNDarray; var _out, where: TNDarray): TNDarray;
      function sinc(x: TNDarray): TNDarray;
      function sinh(x: TNDarray; var _out, where: TNDarray): TNDarray;
      function spacing(x: TNDarray; var _out, where: TNDarray): TNDarray;
      function sqrt(x: TNDarray; var _out, where: TNDarray): TNDarray;
      function subtract(x2, x1: TNDarray; var _out, where: TNDarray): TNDarray;
      function sum(a: TNDarray; axis: TArray<Integer>; dtype: TDtype; var _out: TNDarray; keepdims: PBoolean; initial: TValue): TNDarray;
      function tan(x: TNDarray; var _out, where: TNDarray): TNDarray;
      function tanh(x: TNDarray; var _out, where: TNDarray): TNDarray;
      function trapz(y: TNDarray ; x: TNDarray = nil; dx: Double = 1.0; axis: Integer = -1): Double;
      function true_divide(x1, x2: TNDarray; var _out, where: TNDarray): TNDarray;
      function trunc(x: TNDarray; var _out, where: TNDarray): TNDarray;
      function unwrap(p: TNDarray ; discont : Double = 3.141592653589793; axis: Integer = -1): TNDarray;
      //
      //np.array_creation.gen.cs
      class function empty(shape: Tnp_Shape; dtype : TDtype = nil; order : PChar = nil): TNDarray; overload;
      class function empty(shape : TArray<Integer>): TNDarray; overload; //np.aliases.cs

      class function arange(start : byte ; stop: byte; step  : byte  =  1; dtype  : TDtype  =  nil): TNDarray; overload;
      class Function arange(stop  : byte ;             step  : byte  =  1; dtype  : TDtype  =  nil): TNDarray; overload;
      class Function arange(start : Word ; stop : Word; step  : Word  =  1; dtype  : TDtype  =  nil): TNDarray; overload;
      class Function arange(stop  : Word ;              step  : Word  =  1; dtype  : TDtype  =  nil): TNDarray; overload;
      class Function arange(start : Integer ; stop: Integer; step  : Integer  =  1; dtype  : TDtype  =  nil): TNDarray; overload;
      class Function arange(stop  : Integer ;                step  : Integer  =  1; dtype  : TDtype  =  nil): TNDarray; overload;
      class Function arange(start : int64 ; stop: int64; step  : int64  =  1; dtype  : TDtype  =  nil): TNDarray; overload;
      class Function arange(stop  : Int64 ;              step  : Int64  =  1; dtype  : TDtype  =  nil): TNDarray; overload;
      class Function arange(start : Single ; stop: Single; step  : Single  =  1; dtype  : TDtype  =  nil): TNDarray; overload;
      class Function arange(stop  : Single ;               step  : Single  =  1; dtype  : TDtype  =  nil): TNDarray; overload;
      class Function arange(start : Double ; stop: Double; step  : Double  =  1; dtype  : TDtype  =  nil): TNDarray; overload;
      class Function arange(stop  : Double ;               step  : Double  =  1; dtype  : TDtype  =  nil): TNDarray; overload;

      function asanyarray(a : TNDarray ; dtype  : TDtype  =  nil; order  : pChar = nil): TNDarray; overload;
      function asanyarray<T>(a : TArray<T>; dtype  : TDtype  =  nil; order  : pChar = nil): TNDarray<T>; overload;
      Function asanyarray<T>(a : TArray2D<T>; dtype  : TDtype  =  nil; order  : pChar = nil): TNDarray<T>;overload;

      Function asarray(a : TNDarray ; dtype  : TDtype  =  nil; order  : pChar = nil): TNDarray;overload;
      function asarray<T>(a: TArray<T>; dtype  : TDtype  =  nil; order  : pChar = nil): TNDarray<T>; overload;
      Function asarray<T>(a: TArray2D<T>; dtype  : TDtype  =  nil; order  : pChar = nil): TNDarray<T>; overload;

      function ascontiguousarray(a : TNDarray ; dtype  : TDtype  =  nil): TNDarray; overload;
      function ascontiguousarray<T>(a : TArray<T>; dtype  : TDtype  =  nil): TNDarray<T>; overload;
      Function ascontiguousarray<T>(a : TArray2D<T>; dtype  : TDtype  =  nil): TNDarray<T>;overload;

      function asmatrix(data: TNDarray; dtype: TDtype): TMatrix; overload;
      function asmatrix<T>(data: TArray<T>; dtype: TDtype): TMatrix; overload;
      Function asmatrix<T>(data: TArray2D<T>; dtype: TDtype): TMatrix ;overload;

      procedure chararray(shape : Tnp_Shape ; itemsize  : PInteger  =  nil; unicode  : PBoolean  =  nil; buffer  : PInteger  =  nil; offset  : PInteger  =  nil; strides: TArray<Integer> =  nil; order  : pChar = nil);

      function copy(a : TNDarray ; order  : pChar = nil): TNDarray; overload;
      function copy<T>(a : TArray<T>; order  : pChar = nil): TNDarray<T>;overload;
      Function copy<T>(a : TArray2D<T>; order  : pChar = nil): TNDarray<T>; overload;

      Function diag(v : TNDarray ; k  : Integer  =  0):TNDarray; overload;
      Function diag<T>(v: TArray<T>; k  : Integer  =  0):TNDarray<T>; overload;
      Function diag<T>(v: TArray2D<T>; k  : Integer  =  0): TNDarray<T>; overload;

      function diagflat(v : TNDarray ; k  : Integer  =  0): TNDarray; overload;
      function diagflat<T>(v: TArray<T>; k  : Integer  =  0): TNDarray<T>; overload;
      Function diagflat<T>(v: TArray2D<T>; k  : Integer  =  0):TNDarray<T>; overload;

      function empty_like(prototype: TNDarray ; dtype: TDtype = nil; order: PChar = nil; subok  : Boolean =   true): TNDarray; overload;
      function empty_like<T>(prototype: TArray<T>; dtype  : TDtype  =  nil; order  : pChar  =  nil; subok  : Boolean =   true): TNDarray<T>;  overload;
      Function empty_like<T>(prototype: TArray2D<T>; dtype  : TDtype  =  nil; order  : pChar = nil; subok  : Boolean  =  true): TNDarray<T>;  overload;

      function eye(N : Integer ; M  : PInteger  =  nil; k  : Integer  =  0; dtype  : TDtype  =  nil; order  : pChar = nil): TNDarray;
      procedure fromfile(ffile : string ; dtype  : TDtype  =  nil; count : Integer = -1; sep : string = '');
      function fromstring(sStr: string; dtype  : TDtype  =  nil; count : Integer = -1; sep: string = ''): TNDarray;
      function full(shape : Tnp_Shape ; fill_value: TValue; dtype  : TDtype  =  nil; order  : pChar = nil): TNDarray;

      function full_like(a : TNDarray ; fill_value: TValue; dtype  : TDtype  =  nil; order  : pChar = nil; subok  : Boolean  =  true): TNDarray; overload;
      function full_like<T>(a: TArray<T>; fill_value: TValue; dtype  : TDtype  =  nil; order  : pChar = nil; subok  : Boolean  =  true): TNDarray<T>; overload;
      Function full_like<T>(a:TArray2D<T>; fill_value: TValue; dtype  : TDtype  =  nil; order  : pChar = nil; subok  : Boolean  =  true): TNDarray<T>;  overload;

      function geomspace(start : TNDarray ; stop: TNDarray; num : Integer = 50; endpoint : Boolean  =  true; dtype  : TDtype  =  nil; axis  : Integer  =  0): TNDarray;
      function identity(n : Integer ; dtype  : TDtype  =  nil): TNDarray;

      function linspace(start : TNDarray ; stop: TNDarray ; num  : Integer  =  50; endpoint  : Boolean  =  true; dtype  : TDtype  =  nil; axis  : Integer  =  0): TArray<TValue>; overload;
      Function linspace(start : double ; stop: double; num  : Integer  =  50; endpoint  : Boolean  =  true; dtype  : TDtype  =  nil; axis  : Integer  =  0): TNDarray; overload;

      function loadtxt(fname : string ; dtype  : TDtype  =  nil; comments : TArray<String> =  nil; delimiter  : pChar = nil; converters  : TArray<TVarRec>  =  nil; skiprows  : Integer  =  0; usecols : TArray<Integer>=  nil; unpack  : Boolean  =  false; ndmin  : Integer  =  0; encoding: string  = 'bytes'; max_rows: PInteger =  nil): TNDarray;
      function logspace(start : TNDarray ; stop: TNDarray; num : Integer = 50; endpoint : Boolean = true; base : Double = 10.0; dtype  : TDtype  =  nil; axis  : Integer  =  0): TNDarray;
      procedure mgrid;

      function ones(shape : Tnp_Shape ; dtype  : TDtype  =  nil; order  : pChar = nil): TNDarray; overload;
      function ones(shape : TArray<Integer>): TNDarray; overload ;// np.aliases.cs

      function ones_like(a : TNDarray ; dtype  : TDtype  =  nil; order  : pChar = nil; subok  : Boolean  =  true): TNDarray; overload;
      function ones_like<T>(a: TArray<T>; dtype  : TDtype  =  nil; order  : pChar = nil; subok  : Boolean  =  true): TNDarray<T>; overload;
      Function ones_like<T>(a: TArray2D<T>; dtype  : TDtype  =  nil; order  : pChar = nil; subok  : Boolean  =  true): TNDarray<T>; overload;

      function tri(N : Integer ; M  : PInteger  =  nil; k  : Integer  =  0; dtype  : TDtype  =  nil): TNDarray;

      function tril(m : TNDarray ; k  : Integer  =  0): TNDarray;  overload;
      function tril<T>(m: TArray<T>; k  : Integer  =  0): TNDarray<T>; overload;
      Function tril<T>(m: TArray2D<T>; k  : Integer  =  0):TNDarray<T>; overload;

      function vander(x : TNDarray ; N  : PInteger  =  nil; increasing  : Boolean  =  false): TNDarray; overload;
      function vander<T>(x: TArray<T>; N  : PInteger  =  nil; increasing  : Boolean  =  false): TNDarray<T>;  overload;
      Function vander<T>(x: TArray2D<T>; N  : PInteger  =  nil; increasing  : Boolean  =  false):TNDarray<T>;  overload;

      class function zeros(shape : Tnp_Shape ; dtype  : TDtype  =  nil; order  : pChar = nil): TNDarray; overload;
      class function zeros(shape : TArray<Integer>): TNDarray; overload;// np.aliases.cs

      function zeros_like(a : TNDarray ; dtype  : TDtype  =  nil; order  : pChar = nil; subok  : Boolean  =  true): TNDarray;overload;
      function zeros_like<T>(a: TArray<T>; dtype  : TDtype  =  nil; order  : pChar = nil; subok  : Boolean  =  true): TNDarray<T>; overload;
      Function zeros_like<T>(a: TArray2D<T>; dtype  : TDtype  =  nil; order  : pChar = nil; subok  : Boolean  =  true): TNDarray<T>; overload;
      //
      //np.array.cs
      class function  npArray(obj   : TNDarray;
                        dtype : TDtype   = nil;
                        copy  : PBoolean = nil;
                        order : PChar    = nil;
                        subok : PBoolean = nil ;
                        ndmin : PInteger = nil) : TNDarray;  overload;

      class function  npArray<T>(obj   : TArray<T>;
                           dtype : TDtype   = nil;
                           copy  : PBoolean = nil;
                           order : PChar    = nil;
                           subok : PBoolean = nil ;
                           ndmin : PInteger = nil) : TNDarray<T>; overload;

     class function  npArray<T>(obj    : TArray2D<T>;
                           dtype : TDtype   = nil;
                           copy  : PBoolean = nil;
                           order : PChar    = nil;
                           subok : PBoolean = nil ;
                           ndmin : PInteger = nil) : TNDarray<T>; overload;

     class function  npArray<T>(obj    : TArray3D<T>;
                           dtype : TDtype   = nil;
                           copy  : PBoolean = nil;
                           order : PChar    = nil;
                           subok : PBoolean = nil ;
                           ndmin : PInteger = nil) : TNDarray<T>; overload;

     class function npArray(obj      : TArray<String>;
                      itemsize : PInteger = nil;
                      copy     : PBoolean = nil;
                      unicode  : PBoolean = nil ;
                      order    : PChar    = nil): TNDarray; overload;

     class function npArray(obj   : TArray<TNDarray>;
                      dtype : TDtype   = nil;
                      copy  : PBoolean = nil;
                      order : PChar    = nil;
                      subok : PBoolean = nil ;
                      ndmin : PInteger = nil): TNDarray; overload;

     class Function asarray(scalar : TValue; dtype: TDtype = nil): TNDarray;  overload;
     class function asscalar<T>(a: TNDarray): T;
     //
     //np.random.cs
     function rand(shape: TArray<Integer>): TNDarray;
     function randn(shape: TArray<Integer>): TNDarray;
     //
     //np.linalg.norm.cs
     function norm(x: TNDarray; ord: PInteger; axis: TArray<Integer>; keepdims: PBoolean = nil): TNDarray;  overload;
     function norm(x: TNDarray; ord: PInteger = nil): Double; overload;
     function norm(x: TNDarray; ord: PChar): Double;  overload;
     function norm(x: TNDarray; ord: Constants): Double; overload;
     //
     //np.linspace.cs
     function linspace(start, stop: TNDarray; var step: Double; num:Integer = 50; endpoint:Boolean = true; dtype : TDtype= nil; axis: Integer = 0):TNDarray; overload;
     function linspace(start, stop: Double;   var step: Double; num:Integer = 50; endpoint:Boolean = true; dtype : TDtype= nil; axis: Integer = 0):TNDarray; overload;
     //
     //np.resize.cs
     function resize(a: TNDarray; new_shape: Tnp_Shape): TNDarray;
  end;


implementation
   uses System.Variants;

(********************************//START //np.array_manipulation.gen.cs**)
(************************************************************************)

{ TNumPyArray }

//np.sorting.gen.cs
function TNumPyArray.argmax(a: TNDarray ; axis: PInteger ; var _out : TNDarray): TNDarray;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
    pyargs := TNumPy.ToTuple([a]);

    kwargs := TPyDict.Create;
    if (axis <> nil)         then kwargs['axis']     := ToPython(axis^);
    if (_out <> nil)         then kwargs['out']      := ToPython(_out);
    py := FhModuleNumPy.InvokeMethod('argmax', pyargs, kwargs);

    Result := TNumPy.ToCsharp<TNDarray>(py);

end;

//np.staticstics.gen.cs
function TNumPyArray.amin(a: TNDarray ; axis: TArray<Integer> ; var _out : TNDarray; keepdims : PBoolean; initial : TValue): TNDarray;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
    pyargs := TNumPy.ToTuple([a]);

    kwargs := TPyDict.Create;
    if (axis <> nil)         then kwargs['axis']     := ToPython(TValue.FromArray<Integer>(axis));
    if (_out <> nil)         then kwargs['out']      := ToPython(_out);
    if (keepdims <> nil)     then kwargs['keepdims'] := ToPython(keepdims^);
    if (not initial.IsEmpty) then kwargs['initial']  := ToPython(initial);
    py := FhModuleNumPy.InvokeMethod('amin', pyargs, kwargs);

    Result := TNumPy.ToCsharp<TNDarray>(py);
end;

function TNumPyArray.amax(a: TNDarray ; axis: TArray<Integer> ; var _out : TNDarray; keepdims : PBoolean; initial : TValue): TNDarray;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
    pyargs := TNumPy.ToTuple([a]);

    kwargs := TPyDict.Create;
    if (axis <> nil)         then kwargs['axis']     := ToPython(TValue.FromArray<Integer>(axis));
    if (_out <> nil)         then kwargs['out']      := ToPython(_out);
    if (keepdims <> nil)     then kwargs['keepdims'] := ToPython(keepdims^);
    if (not initial.IsEmpty) then kwargs['initial']  := ToPython(initial);
    py := FhModuleNumPy.InvokeMethod('amax', pyargs, kwargs);

    Result := TNumPy.ToCsharp<TNDarray>(py);

end;

function TNumPyArray.min(a: TNDarray ; axis: TArray<Integer> ; var _out : TNDarray; keepdims : PBoolean; initial : TValue): TNDarray;
begin
    Result := amin(a,axis,_out,keepdims,initial);
end;

function TNumPyArray.max(a: TNDarray ; axis: TArray<Integer> ; var _out : TNDarray; keepdims : PBoolean; initial : TValue): TNDarray;
begin
    Result := amax(a,axis,_out,keepdims,initial);
end;

class function TNumPyArray.reshape(a: TNDarray; newshape: TArray<Integer>): TNDarray;
begin
    Result := reshape(a, Tnp_Shape.Create(newshape));
end;

class function TNumPyArray.reshape(a: TNDarray; newshape: Tnp_Shape; order : PChar = nil): TNDarray;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
    pyargs := TNumPy.ToTuple([a, TValue.FromShape(newshape)]);

    kwargs := TPyDict.Create;
    if (order<>nil) then kwargs['order'] :=  TNumPy.ToPython(order);
    py := FhModuleNumPy.InvokeMethod('reshape', pyargs, kwargs);

    Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Procedure TNumPyArray.copyto(dst : TNDarray  ; src : TNDarray  ; casting : string = 'same_kind'; where : TNDarray = nil);
var
  pyargs : TPyTuple;
  kwargs : TPyDict;

begin

    pyargs := TNumPy.ToTuple([dst,src]);
    kwargs := TPyDict.Create;
    if (casting<>'same_kind') then kwargs['casting'] := ToPython(casting);
    if (where<>nil) then kwargs['where'] := ToPython(@where);
    FhModuleNumPy.InvokeMethod('copyto', pyargs, kwargs);
end;


Procedure TNumPyArray.copyto(dst : TNDarray  ; src : TNDarray  ; casting : string = 'same_kind'; where : TArray<Boolean> = []);
var
  pyargs : TPyTuple;
  kwargs : TPyDict;

begin

    pyargs := TNumPy.ToTuple([dst,src]);
    kwargs := TPyDict.Create;
    if (casting<>'same_kind') then kwargs['casting'] := ToPython(casting);
    if (where<>nil) then kwargs['where'] := ToPython(@where);
    FhModuleNumPy.InvokeMethod('copyto', pyargs, kwargs);
end;


Function TNumPyArray.ravel(a : TNDarray  ; order : PChar = nil):TNDarray;
var
  pyargs : TPyTuple;
  kwargs : TPyDict;
  py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([a]);
    kwargs := TPyDict.Create;
    if (order<>nil) then kwargs['order'] := ToPython(order);
    py := FhModuleNumPy.InvokeMethod('ravel', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray>(py);
end;


Function TNumPyArray.flatten(order : PChar = nil):TNDarray;
var
  pyargs : TPyTuple;
  kwargs : TPyDict;
  py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([]);
    kwargs := TPyDict.Create;
    if (order<>nil) then kwargs['order'] := ToPython(order);
    py := FhModuleNumPy.InvokeMethod('flatten', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray>(py);
end;


Function TNumPyArray.moveaxis(a : TNDarray  ; source : TArray<Integer>  ; destination : TArray<Integer> ):TNDarray;
var
  pyargs : TPyTuple;
  kwargs : TPyDict;
  py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([a, TValue.FromArray<Integer>(source),TValue.FromArray<Integer>(destination)]);
    kwargs := TPyDict.Create;
    py := FhModuleNumPy.InvokeMethod('moveaxis', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray>(py);
end;


Function TNumPyArray.rollaxis(a : TNDarray  ; axis : Integer ; start : Integer = 0):TNDarray;
var
  pyargs : TPyTuple;
  kwargs : TPyDict;
  py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([a,axis]);
    kwargs := TPyDict.Create;
    if (start<>0) then kwargs['start'] := ToPython(start);
    py := FhModuleNumPy.InvokeMethod('rollaxis', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray>(py);
end;


Function TNumPyArray.swapaxes(a : TNDarray  ; axis1 : Integer ; axis2 : Integer):TNDarray;
var
  pyargs : TPyTuple;
  kwargs : TPyDict;
  py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([a,axis1,axis2]);
    kwargs := TPyDict.Create;
    py := FhModuleNumPy.InvokeMethod('swapaxes', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray>(py);
end;


Function TNumPyArray.transpose(a : TNDarray  ; axes : TArray<Integer> = []):TNDarray;
var
  pyargs : TPyTuple;
  kwargs : TPyDict;
  py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([a]);
    kwargs := TPyDict.Create;
    if (axes<>nil) then kwargs['axes'] := ToPython(TValue.FromArray<Integer>(axes));
    py := FhModuleNumPy.InvokeMethod('transpose', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray>(py);
end;


Function TNumPyArray.atleast_1d( arys : TArray<TNDarray> ):TNDarray;
var
  pyargs : TPyTuple;
  kwargs : TPyDict;
  py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([TValue.FromArray<TNDarray>(arys)]);
    kwargs := TPyDict.Create;
    py := FhModuleNumPy.InvokeMethod('atleast_1d', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray>(py);
end;


Function  TNumPyArray.atleast_2d( arys : TArray<TNDarray> ):TNDarray;
var
  pyargs : TPyTuple;
  kwargs : TPyDict;
  py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([TValue.FromArray<TNDarray>(arys)]);
    kwargs := TPyDict.Create;
    py := FhModuleNumPy.InvokeMethod('atleast_2d', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray>(py);
end;


Function TNumPyArray.atleast_3d( arys : TArray<TNDarray> ):TNDarray;
var
  pyargs : TPyTuple;
  kwargs : TPyDict;
  py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([TValue.FromArray<TNDarray>(arys)]);
    kwargs := TPyDict.Create;
    py := FhModuleNumPy.InvokeMethod('atleast_3d', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray>(py);
end;


Function TNumPyArray.broadcast(in2 : TNDarray  ; in1 : TNDarray ):TNDarray;
var
  pyargs : TPyTuple;
  kwargs : TPyDict;
  py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([in2,in1]);
    kwargs := TPyDict.Create;
    py := FhModuleNumPy.InvokeMethod('broadcast', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray>(py);
end;


Function TNumPyArray.broadcast_to(arr : TNDarray  ; shape : Tnp_Shape  ; subok : Boolean = false):TNDarray;
var
  pyargs : TPyTuple;
  kwargs : TPyDict;
  py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([arr, TValue.FromShape(shape)]);
    kwargs := TPyDict.Create;
    if (subok<>false) then kwargs['subok'] := ToPython(subok);
    py := FhModuleNumPy.InvokeMethod('broadcast_to', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray>(py);
end;


Function TNumPyArray.broadcast_arrays(args : TArray<TNDarray>  ; subok : PBoolean = nil): TArray<TNDarray>;
var
  pyargs : TPyTuple;
  kwargs : TPyDict;
  py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([TValue.FromArray<TNDarray>(args)]);
    kwargs := TPyDict.Create;
    if (subok<>nil) then kwargs['subok'] := ToPython(subok^);
    py := FhModuleNumPy.InvokeMethod('broadcast_arrays', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TArray<TNDarray>>(py);
end;


Function TNumPyArray.expand_dims(a : TNDarray  ; axis : Integer):TNDarray;
var
  pyargs : TPyTuple;
  kwargs : TPyDict;
  py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([a,axis]);
    kwargs := TPyDict.Create;
    py := FhModuleNumPy.InvokeMethod('expand_dims', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray>(py);
end;


Function TNumPyArray.squeeze(a : TNDarray  ; axis : TArray<Integer> = []):TNDarray;
var
  pyargs : TPyTuple;
  kwargs : TPyDict;
  py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([a]);
    kwargs := TPyDict.Create;
    if (axis<>nil) then kwargs['axis'] := ToPython(TValue.FromArray<Integer>(axis));
    py := FhModuleNumPy.InvokeMethod('squeeze', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray>(py);
end;


Function TNumPyArray.asfarray(a : TNDarray  ; dtype : TDtype = nil): TNDarray;
var
  pyargs : TPyTuple;
  kwargs : TPyDict;
  py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([a]);
    kwargs := TPyDict.Create;
    if (dtype<>nil) then kwargs['dtype'] := ToPython(dtype);
    py := FhModuleNumPy.InvokeMethod('asfarray', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray>(py);
end;


Function TNumPyArray.asfortranarray(a : TNDarray  ; dtype : TDtype = nil):TNDarray;
var
  pyargs : TPyTuple;
  kwargs : TPyDict;
  py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([a]);
    kwargs := TPyDict.Create;
    if (dtype<>nil) then kwargs['dtype'] := ToPython(dtype);
    py := FhModuleNumPy.InvokeMethod('asfortranarray', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray>(py);
end;


Function TNumPyArray.asarray_chkfinite(a : TNDarray  ; dtype : TDtype = nil; order : PChar = nil): TNDarray;
var
  pyargs : TPyTuple;
  kwargs : TPyDict;
  py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([a]);
    kwargs := TPyDict.Create;
    if (dtype<>nil) then kwargs['dtype'] := ToPython(dtype);
    if (order<>nil) then kwargs['order'] := ToPython(order);
    py := FhModuleNumPy.InvokeMethod('asarray_chkfinite', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray>(py);
end;


Function TNumPyArray.require(a : TNDarray  ; dtype : TDtype  ; requirements : TArray<string>= []):TNDarray;
var
  pyargs : TPyTuple;
  kwargs : TPyDict;
  py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([a,dtype]);
    kwargs := TPyDict.Create;
    if (requirements<>nil) then kwargs['requirements'] := ToPython(TValue.FromArray<string>(requirements));
    py := FhModuleNumPy.InvokeMethod('require', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray>(py);
end;


Function TNumPyArray.concatenate(arys : TArray<TNDarray>  ; axis : Integer ; var _out : TNDarray):TNDarray;
var
  pyargs : TPyTuple;
  kwargs : TPyDict;
  py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([TValue.FromArray<TNDarray>(arys)]);
    kwargs := TPyDict.Create;
    if (axis<>0) then kwargs['axis'] := ToPython(axis);
    if (_out<>nil) then kwargs['out'] := ToPython(_out);
    py := FhModuleNumPy.InvokeMethod('concatenate', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray>(py);
end;


Function TNumPyArray.stack(arrays : TArray<TNDarray>  ; axis : Integer ; var _out : TNDarray):TNDarray;
var
  pyargs : TPyTuple;
  kwargs : TPyDict;
  py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([TValue.FromArray<TNDarray>(arrays)]);
    kwargs := TPyDict.Create;
    if (axis<>0) then kwargs['axis'] := ToPython(axis);
    if (_out<>nil) then kwargs['out'] := ToPython(_out);
    py := FhModuleNumPy.InvokeMethod('stack', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray>(py);
end;


Function TNumPyArray.column_stack( tup : TArray<TNDarray> ):TNDarray;
var
  pyargs : TPyTuple;
  kwargs : TPyDict;
  py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([TValue.FromArray<TNDarray>(tup)]);
    kwargs := TPyDict.Create;
    py := FhModuleNumPy.InvokeMethod('column_stack', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray>(py);
end;


Function TNumPyArray.dstack( tup : TArray<TNDarray> ):TNDarray;
var
  pyargs : TPyTuple;
  kwargs : TPyDict;
  py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([TValue.FromArray<TNDarray>(tup)]);
    kwargs := TPyDict.Create;
    py := FhModuleNumPy.InvokeMethod('dstack', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray>(py);
end;


Function TNumPyArray.hstack( tup : TArray<TNDarray> ):TNDarray;
var
  pyargs : TPyTuple;
  kwargs : TPyDict;
  py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([TValue.FromArray<TNDarray>(tup)]);
    kwargs := TPyDict.Create;
    py := FhModuleNumPy.InvokeMethod('hstack', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray>(py);
end;


Function TNumPyArray.vstack( tup : TArray<TNDarray> ):TNDarray;
var
  pyargs : TPyTuple;
  kwargs : TPyDict;
  py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([TValue.FromArray<TNDarray>(tup)]);
    kwargs := TPyDict.Create;
    py := FhModuleNumPy.InvokeMethod('vstack', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray>(py);
end;

(*
Function block(nested list of array_like or scalars (but tuples : not ) arrays): TNDarray;
var
  pyargs : TPyTuple;
  kwargs : TPyDict;
  py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([arrays]);
    kwargs := TPyDict.Create;
    py := FhModuleNumPy.InvokeMethod('block', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray>(py);
end;
*)


Function TNumPyArray.split(ary : TNDarray  ; indices_or_sections : TArray<Integer>; axis : Integer = 0): TArray<TNDarray>;
var
  pyargs : TPyTuple;
  kwargs : TPyDict;
  py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([ary,TValue.FromArray<Integer>(indices_or_sections)]);
    kwargs := TPyDict.Create;
    if (axis<>0) then kwargs['axis'] := ToPython(axis);
    py := FhModuleNumPy.InvokeMethod('split', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TArray<TNDarray>>(py);
end;


Function TNumPyArray.tile(A : TNDarray  ; reps : TNDarray ): TNDarray;
var
  pyargs : TPyTuple;
  kwargs : TPyDict;
  py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([A,reps]);
    kwargs := TPyDict.Create;
    py := FhModuleNumPy.InvokeMethod('tile', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray>(py);
end;


Function TNumPyArray.&repeat(a : TNDarray  ; repeats : TArray<Integer>  ; axis : PInteger = nil): TNDarray;
var
  pyargs : TPyTuple;
  kwargs : TPyDict;
  py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([a,TValue.FromArray<Integer>(repeats)]);
    kwargs := TPyDict.Create;
    if (axis<>nil) then kwargs['axis'] := ToPython(axis^);
    py := FhModuleNumPy.InvokeMethod('repeat', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray>(py);
end;


Function TNumPyArray.delete(arr : TNDarray  ; obj : Tnp_Slice  ; axis : PInteger = nil): TNDarray;
var
  pyargs : TPyTuple;
  kwargs : TPyDict;
  py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([arr,TValue.FromSlice(obj)]);
    kwargs := TPyDict.Create;
    if (axis<>nil) then kwargs['axis'] := ToPython(axis^);
    py := FhModuleNumPy.InvokeMethod('delete', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray>(py);
end;


Function TNumPyArray.insert(arr : TNDarray  ; obj : Integer = 0; values : TNDarray  = nil; axis : PInteger = nil): TNDarray;
var
  pyargs : TPyTuple;
  kwargs : TPyDict;
  py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([arr]);
    kwargs := TPyDict.Create;
    if (obj<>0) then kwargs['obj'] := ToPython(obj);
    if (values<>nil) then kwargs['values'] := ToPython(values);
    if (axis<>nil) then kwargs['axis'] := ToPython(axis^);
    py := FhModuleNumPy.InvokeMethod('insert', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray>(py);
end;


Function TNumPyArray.append(arr : TNDarray  ; values : TNDarray  ; axis : PInteger = nil): TNDarray;
var
  pyargs : TPyTuple;
  kwargs : TPyDict;
  py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([arr,values]);
    kwargs := TPyDict.Create;
    if (axis<>nil) then kwargs['axis'] := ToPython(axis^);
    py := FhModuleNumPy.InvokeMethod('append', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray>(py);
end;


Function TNumPyArray.trim_zeros(filt : TNDarray  ; trim : string = 'fb'): TNDarray;
var
  pyargs : TPyTuple;
  kwargs : TPyDict;
  py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([filt]);
    kwargs := TPyDict.Create;
    if (trim<>'fb') then kwargs['trim'] := ToPython(trim);
    py := FhModuleNumPy.InvokeMethod('trim_zeros', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray>(py);
end;


Function TNumPyArray.unique(ar : TNDarray  ; axis : PInteger = nil): TNDarray;
var
  pyargs : TPyTuple;
  kwargs : TPyDict;
  py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([ar]);
    kwargs := TPyDict.Create;
    if (axis<>nil) then kwargs['axis'] := ToPython(axis^);
    py := FhModuleNumPy.InvokeMethod('unique', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray>(py);
end;


Function TNumPyArray.unique(ar : TNDarray  ; return_index: Boolean= False ; return_inverse: Boolean=False; return_counts: Boolean=false; axis : PInteger = nil):TArray<TNDarray>;
var
  pyargs : TPyTuple;
  kwargs : TPyDict;
  py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([ar]);
    kwargs := TPyDict.Create;
    if (return_index<>False) then kwargs['return_index'] := ToPython(return_index);
    if (return_inverse<>False) then kwargs['return_inverse'] := ToPython(return_inverse);
    if (return_counts<>False) then kwargs['return_counts'] := ToPython(return_counts);
    if (axis<>nil) then kwargs['axis'] := ToPython(axis^);
    py := FhModuleNumPy.InvokeMethod('unique', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TArray<TNDarray>>(py);
end;


Function TNumPyArray.flip(m : TNDarray  ; axis : TArray<Integer> = []): TNDarray;
var
  pyargs : TPyTuple;
  kwargs : TPyDict;
  py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([m]);
    kwargs := TPyDict.Create;
    if (axis<>nil) then kwargs['axis'] := ToPython(TValue.FromArray<Integer>(axis));
    py := FhModuleNumPy.InvokeMethod('flip', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray>(py);
end;


Function TNumPyArray.fliplr(m : TNDarray ):TNDarray;
var
  pyargs : TPyTuple;
  kwargs : TPyDict;
  py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([m]);
    kwargs := TPyDict.Create;
    py := FhModuleNumPy.InvokeMethod('fliplr', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray>(py);
end;


Function TNumPyArray.flipud(m : TNDarray ):TNDarray;
var
  pyargs : TPyTuple;
  kwargs : TPyDict;
  py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([m]);
    kwargs := TPyDict.Create;
    py := FhModuleNumPy.InvokeMethod('flipud', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray>(py);
end;


Function TNumPyArray.roll(a : TNDarray  ; shift : TArray<Integer>  ; axis : TArray<Integer> = []):TNDarray;
var
  pyargs : TPyTuple;
  kwargs : TPyDict;
  py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([a, TValue.FromArray<Integer>(shift)]);
    kwargs := TPyDict.Create;
    if (axis<>nil) then kwargs['axis'] := ToPython(TValue.FromArray<Integer>(axis));
    py := FhModuleNumPy.InvokeMethod('roll', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray>(py);
end;


Function TNumPyArray.rot90(m : TNDarray  ; k : Integer = 1; axes : TArray<Integer> = []): TNDarray;
var
  pyargs : TPyTuple;
  kwargs : TPyDict;
  py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([m]);
    kwargs := TPyDict.Create;
    if (k<>1) then kwargs['k'] := ToPython(k);
    if (axes<>nil) then kwargs['axes'] := ToPython(TValue.FromArray<Integer>(axes));
    py := FhModuleNumPy.InvokeMethod('rot90', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray>(py);
end;

(********************************//END //np.array_manipulation.gen.cs**)
(************************************************************************)

(*********************************** //START //np.math.gen.cs ***********)
(************************************************************************)

function TNumPyArray.maximum(x2, x1: TNDarray; var _out : TNDarray; var where : TNDarray): TNDarray;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple([x2,x1]);
  kwargs := TPyDict.Create;
  if (_out <> nil) then kwargs['out'] := TNumPy.ToPython(_out);
  if (where <> nil) then kwargs['where'] := TNumPy.ToPython(where);
  py := FhModuleNumPy.InvokeMethod('maximum', pyargs, kwargs);

  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

function TNumPyArray.maximum(x2, x1: TNDarray; var _out : TNDarray): TNDarray;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple([x2,x1]);
  kwargs := TPyDict.Create;
  if (_out <> nil) then kwargs['out'] := TNumPy.ToPython(_out);
  py := FhModuleNumPy.InvokeMethod('maximum', pyargs, kwargs);

  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

function TNumPyArray.maximum(x2, x1: TNDarray): TNDarray;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple([x2,x1]);
  kwargs := TPyDict.Create;

  py := FhModuleNumPy.InvokeMethod('maximum', pyargs, kwargs);

  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

function TNumPyArray.square(x: TNDarray; var _out : TNDarray; var where : TNDarray): TNDarray;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple([x]);
  kwargs := TPyDict.Create;
  if (_out <> nil) then kwargs['out'] := TNumPy.ToPython(_out);
  if (where <> nil) then kwargs['where'] := TNumPy.ToPython(where);
  py := FhModuleNumPy.InvokeMethod('square', pyargs, kwargs);

  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

function TNumPyArray.square(x: TNDarray; var _out : TNDarray): TNDarray;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple([x]);
  kwargs := TPyDict.Create;
  if (_out <> nil) then kwargs['out'] := TNumPy.ToPython(_out);

  py := FhModuleNumPy.InvokeMethod('square', pyargs, kwargs);

  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

function TNumPyArray.square(x: TNDarray): TNDarray;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple([x]);
  kwargs := TPyDict.Create;

  py := FhModuleNumPy.InvokeMethod('square', pyargs, kwargs);

  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNumPyArray.sin(x: TNDarray ; var _out : TNDarray; var where : TNDarray):TNDarray;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([x]);
    kwargs := TPyDict.Create;
    if (_out <> nil)  then kwargs['out']   := ToPython(_out);
    if (where <> nil) then kwargs['where'] := ToPython(where);
    py := FhModuleNumPy.InvokeMethod('sin', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNumPyArray.cos(x: TNDarray ; var _out : TNDarray; var where : TNDarray):TNDarray;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([x]);
    kwargs := TPyDict.Create;
    if (_out <> nil)  then kwargs['out']   := ToPython(_out);
    if (where <> nil) then kwargs['where'] := ToPython(where);
    py := FhModuleNumPy.InvokeMethod('cos', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNumPyArray.tan(x: TNDarray ; var _out : TNDarray; var where : TNDarray):TNDarray;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([x]);
    kwargs := TPyDict.Create;
    if (_out <> nil)  then kwargs['out']    := ToPython(_out);
    if (where <> nil) then kwargs['where']  := ToPython(where);
    py := FhModuleNumPy.InvokeMethod('tan', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNumPyArray.arcsin(x: TNDarray ; var _out : TNDarray; var where : TNDarray):TNDarray;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([x]);
    kwargs := TPyDict.Create;
    if (_out <> nil)  then kwargs['out']   := ToPython(_out);
    if (where <> nil) then kwargs['where'] := ToPython(where);
    py := FhModuleNumPy.InvokeMethod('arcsin', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNumPyArray.arccos(x: TNDarray ; var _out : TNDarray; var where : TNDarray):TNDarray;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([x]);
    kwargs := TPyDict.Create;
    if (_out <> nil)   then kwargs['out']   := ToPython(_out);
    if (where <> nil)  then kwargs['where'] := ToPython(where);
    py := FhModuleNumPy.InvokeMethod('arccos', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNumPyArray.arctan(x: TNDarray ; var _out : TNDarray; var where : TNDarray):TNDarray;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([x]);
    kwargs := TPyDict.Create;
    if (_out <> nil)  then kwargs['out']   := ToPython(_out);
    if (where <> nil) then kwargs['where'] := ToPython(where);
    py := FhModuleNumPy.InvokeMethod('arctan', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNumPyArray.hypot(x2: TNDarray ; x1: TNDarray ; var _out : TNDarray; var where : TNDarray):TNDarray;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([x2,x1]);
    kwargs := TPyDict.Create;
    if (_out <> nil)   then kwargs['out']  := ToPython(_out);
    if (where <> nil) then kwargs['where'] := ToPython(where);
    py := FhModuleNumPy.InvokeMethod('hypot', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNumPyArray.arctan2(x1: TNDarray ; x2: TNDarray ; var _out : TNDarray; var where : TNDarray):TNDarray;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([x1,x2]);
    kwargs := TPyDict.Create;
    if (_out <> nil)  then kwargs['out']   := ToPython(_out);
    if (where <> nil) then kwargs['where'] := ToPython(where);
    py := FhModuleNumPy.InvokeMethod('arctan2', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNumPyArray.degrees(x: TNDarray ; var _out : TNDarray; var where : TNDarray):TNDarray;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([x]);
    kwargs := TPyDict.Create;
    if (_out <> nil)  then kwargs['out']   := ToPython(_out);
    if (where <> nil) then kwargs['where'] := ToPython(where);
    py := FhModuleNumPy.InvokeMethod('degrees', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNumPyArray.radians(x: TNDarray ; var _out : TNDarray; var where : TNDarray):TNDarray;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([x]);
    kwargs := TPyDict.Create;
    if (_out <> nil) then kwargs['out'] := ToPython(_out);
    if (where <> nil) then kwargs['where'] := ToPython(where);
    py := FhModuleNumPy.InvokeMethod('radians', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNumPyArray.unwrap(p: TNDarray ; discont : Double = 3.141592653589793; axis: Integer = -1):TNDarray;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([p]);
    kwargs := TPyDict.Create;
    if (discont <> 3.141592653589793) then kwargs['discont'] := ToPython(discont);
    if (axis <> -1)                   then kwargs['axis']    := ToPython(axis);
    py := FhModuleNumPy.InvokeMethod('unwrap', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNumPyArray.deg2rad(x: TNDarray ; var _out : TNDarray; var where : TNDarray):TNDarray;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([x]);
    kwargs := TPyDict.Create;
    if (_out <> nil)  then kwargs['out']   := ToPython(_out);
    if (where <> nil) then kwargs['where'] := ToPython(where);
    py := FhModuleNumPy.InvokeMethod('deg2rad', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNumPyArray.rad2deg(x: TNDarray ; var _out : TNDarray; var where : TNDarray):TNDarray;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([x]);
    kwargs := TPyDict.Create;
    if (_out <> nil)  then kwargs['out']   := ToPython(_out);
    if (where <> nil) then kwargs['where'] := ToPython(where);
    py := FhModuleNumPy.InvokeMethod('rad2deg', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNumPyArray.sinh(x: TNDarray ; var _out : TNDarray; var where : TNDarray):TNDarray;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([x]);
    kwargs := TPyDict.Create;
    if (_out <> nil)  then kwargs['out']   := ToPython(_out);
    if (where <> nil) then kwargs['where'] := ToPython(where);
    py := FhModuleNumPy.InvokeMethod('sinh', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNumPyArray.cosh(x: TNDarray ; var _out : TNDarray; var where : TNDarray):TNDarray;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([x]);
    kwargs := TPyDict.Create;
    if (_out <> nil)  then kwargs['out']   := ToPython(_out);
    if (where <> nil) then kwargs['where'] := ToPython(where);
    py := FhModuleNumPy.InvokeMethod('cosh', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNumPyArray.tanh(x: TNDarray ; var _out : TNDarray; var where : TNDarray):TNDarray;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([x]);
    kwargs := TPyDict.Create;
    if (_out <> nil)  then kwargs['out']   := ToPython(_out);
    if (where <> nil) then kwargs['where'] := ToPython(where);
    py := FhModuleNumPy.InvokeMethod('tanh', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNumPyArray.arcsinh(x: TNDarray ; var _out : TNDarray; var where : TNDarray):TNDarray;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([x]);
    kwargs := TPyDict.Create;
    if (_out <> nil)  then kwargs['out'] := ToPython(_out);
    if (where <> nil) then kwargs['where'] := ToPython(where);
    py := FhModuleNumPy.InvokeMethod('arcsinh', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNumPyArray.arccosh(x: TNDarray ; var _out : TNDarray; var where : TNDarray):TNDarray;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([x]);
    kwargs := TPyDict.Create;
    if (_out <> nil) then kwargs['out'] := ToPython(_out);
    if (where <> nil) then kwargs['where'] := ToPython(where);
    py := FhModuleNumPy.InvokeMethod('arccosh', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNumPyArray.arctanh(x: TNDarray ; var _out : TNDarray; var where : TNDarray):TNDarray;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([x]);
    kwargs := TPyDict.Create;
    if (_out <> nil) then kwargs['out'] := ToPython(_out);
    if (where <> nil) then kwargs['where'] := ToPython(where);
    py := FhModuleNumPy.InvokeMethod('arctanh', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNumPyArray.around(a: TNDarray ; decimals : Integer ; var _out : TNDarray ):TNDarray;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([a]);
    kwargs := TPyDict.Create;
    if (decimals <> 0) then kwargs['decimals'] := ToPython(decimals);
    if (_out <> nil) then kwargs['out'] := ToPython(_out);
    py := FhModuleNumPy.InvokeMethod('around', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNumPyArray.rint(x: TNDarray ; var _out : TNDarray; var where : TNDarray):TNDarray;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([x]);
    kwargs := TPyDict.Create;
    if (_out <> nil) then kwargs['out'] := ToPython(_out);
    if (where <> nil) then kwargs['where'] := ToPython(where);
    py := FhModuleNumPy.InvokeMethod('rint', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNumPyArray.fix(x: TNDarray ; y: TNDarray = nil):TNDarray;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([x]);
    kwargs := TPyDict.Create;
    if (y <> nil) then kwargs['y'] := ToPython(y);
    py := FhModuleNumPy.InvokeMethod('fix', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNumPyArray.floor(x: TNDarray ; var _out : TNDarray; var where : TNDarray):TNDarray;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([x]);
    kwargs := TPyDict.Create;
    if (_out <> nil) then kwargs['out'] := ToPython(_out);
    if (where <> nil) then kwargs['where'] := ToPython(where);
    py := FhModuleNumPy.InvokeMethod('floor', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNumPyArray.ceil(x: TNDarray ; var _out : TNDarray; var where : TNDarray):TNDarray;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([x]);
    kwargs := TPyDict.Create;
    if (_out <> nil) then kwargs['out'] := ToPython(_out);
    if (where <> nil) then kwargs['where'] := ToPython(where);
    py := FhModuleNumPy.InvokeMethod('ceil', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNumPyArray.trunc(x: TNDarray ; var _out : TNDarray; var where : TNDarray):TNDarray;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([x]);
    kwargs := TPyDict.Create;
    if (_out <> nil) then kwargs['out'] := ToPython(_out);
    if (where <> nil) then kwargs['where'] := ToPython(where);
    py := FhModuleNumPy.InvokeMethod('trunc', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNumPyArray.prod(a: TNDarray ; axis: TArray<Integer> ; dtype: TDtype  ; var _out : TNDarray; keepdims : PBoolean ; initial: TValue):TNDarray;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([a]);
    kwargs := TPyDict.Create;
    if (axis <> nil) then kwargs['axis'] := ToPython(TValue.FromArray<Integer>(axis));
    if (dtype <> nil) then kwargs['dtype'] := ToPython(dtype);
    if (_out <> nil) then kwargs['out'] := ToPython(_out);
    if (keepdims <> nil) then kwargs['keepdims'] := ToPython(keepdims^);
    if (not initial.IsEmpty) then kwargs['initial'] := ToPython(initial);
    py := FhModuleNumPy.InvokeMethod('prod', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNumPyArray.sum(a: TNDarray ; axis: TArray<Integer> ; dtype: TDtype ; var _out : TNDarray; keepdims : PBoolean; initial : TValue):TNDarray;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([a]);
    kwargs := TPyDict.Create;
    if (axis <> nil) then kwargs['axis'] := ToPython(TValue.FromArray<Integer>(axis));
    if (dtype <> nil) then kwargs['dtype'] := ToPython(dtype);
    if (_out <> nil) then kwargs['out'] := ToPython(_out);
    if (keepdims <> nil) then kwargs['keepdims'] := ToPython(keepdims^);
    if (not initial.IsEmpty) then kwargs['initial'] := ToPython(initial);
    py := FhModuleNumPy.InvokeMethod('sum', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNumPyArray.nanprod(a: TNDarray ; axis: TArray<Integer> ; dtype: TDtype  ; var _out : TNDarray; keepdims : PBoolean = nil):TNDarray;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([a]);
    kwargs := TPyDict.Create;
    if (axis <> nil) then kwargs['axis'] := ToPython(TValue.FromArray<Integer>(axis));
    if (dtype <> nil) then kwargs['dtype'] := ToPython(dtype);
    if (_out <> nil) then kwargs['out'] := ToPython(_out);
    if (keepdims <> nil) then kwargs['keepdims'] := ToPython(keepdims^);
    py := FhModuleNumPy.InvokeMethod('nanprod', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNumPyArray.nansum(a: TNDarray ; axis: TArray<Integer> ; dtype: TDtype  ; var _out : TNDarray; keepdims : PBoolean = nil):TNDarray;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([a]);
    kwargs := TPyDict.Create;
    if (axis <> nil) then kwargs['axis'] := ToPython(TValue.FromArray<Integer>(axis));
    if (dtype <> nil) then kwargs['dtype'] := ToPython(dtype);
    if (_out <> nil) then kwargs['out'] := ToPython(_out);
    if (keepdims <> nil) then kwargs['keepdims'] := ToPython(keepdims^);
    py := FhModuleNumPy.InvokeMethod('nansum', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNumPyArray.cumprod(a: TNDarray ; axis: PInteger ; dtype: TDtype  ; var _out : TNDarray):TNDarray;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([a]);
    kwargs := TPyDict.Create;
    if (axis <> nil) then kwargs['axis'] := ToPython(axis^);
    if (dtype <> nil) then kwargs['dtype'] := ToPython(dtype);
    if (_out <> nil) then kwargs['out'] := ToPython(_out);
    py := FhModuleNumPy.InvokeMethod('cumprod', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNumPyArray.cumsum(a: TNDarray ; axis: PInteger ; dtype: TDtype ; var _out: TNDarray):TNDarray;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([a]);
    kwargs := TPyDict.Create;
    if (axis <> nil) then kwargs['axis'] := ToPython(axis^);
    if (dtype <> nil) then kwargs['dtype'] := ToPython(dtype);
    if (_out <> nil) then kwargs['out'] := ToPython(_out);
    py := FhModuleNumPy.InvokeMethod('cumsum', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNumPyArray.nancumprod(a: TNDarray ; axis: PInteger ; dtype: TDtype ; var _out: TNDarray):TNDarray;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([a]);
    kwargs := TPyDict.Create;
    if (axis <> nil) then kwargs['axis'] := ToPython(axis^);
    if (dtype <> nil) then kwargs['dtype'] := ToPython(dtype);
    if (_out <> nil) then kwargs['out'] := ToPython(_out);
    py := FhModuleNumPy.InvokeMethod('nancumprod', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNumPyArray.nancumsum(a: TNDarray ; axis : PInteger ; dtype: TDtype ; var _out: TNDarray ):TNDarray;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([a]);
    kwargs := TPyDict.Create;
    if (axis <> nil) then kwargs['axis'] := ToPython(axis^);
    if (dtype <> nil) then kwargs['dtype'] := ToPython(dtype);
    if (_out <> nil) then kwargs['out'] := ToPython(_out);
    py := FhModuleNumPy.InvokeMethod('nancumsum', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNumPyArray.diff(a: TNDarray ; n: Integer = 1; axis: INteger = -1; append: TNDarray = nil; prepend: TNDarray = nil):TNDarray;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([a]);
    kwargs := TPyDict.Create;
    if (n <> 1) then kwargs['n'] := ToPython(n);
    if (axis <> -1) then kwargs['axis'] := ToPython(axis);
    if (append <> nil) then kwargs['append'] := ToPython(append);
    if (prepend <> nil) then kwargs['prepend'] := ToPython(prepend);
    py := FhModuleNumPy.InvokeMethod('diff', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNumPyArray.ediff1d(ary: TNDarray ; to_end: TNDarray = nil; to_begin: TNDarray = nil):TNDarray;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([ary]);
    kwargs := TPyDict.Create;
    if (to_end <> nil) then kwargs['to_end'] := ToPython(to_end);
    if (to_begin <> nil) then kwargs['to_begin'] := ToPython(to_begin);
    py := FhModuleNumPy.InvokeMethod('ediff1d', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNumPyArray.gradient(f: TNDarray ; vararg : TNDarray = nil; edge_order : PInteger = nil; axis: TArray<Integer> = []):TNDarray;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([f]);
    kwargs := TPyDict.Create;
    if (vararg<> nil) then kwargs['varargs'] := ToPython(vararg);
    if (edge_order <> nil) then kwargs['edge_order'] := ToPython(edge_order^);
    if (axis <> nil) then kwargs['axis'] := ToPython(TValue.FromArray<Integer>(axis));
    py := FhModuleNumPy.InvokeMethod('gradient', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNumPyArray.cross(a: TNDarray ; b: TNDarray ; axisa: Integer = -1; axisb: INteger = -1; axisc: Integer = -1; axis: PInteger = nil):TNDarray;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([a,b]);
    kwargs := TPyDict.Create;
    if (axisa <> -1) then kwargs['axisa'] := ToPython(axisa);
    if (axisb <> -1) then kwargs['axisb'] := ToPython(axisb);
    if (axisc <> -1) then kwargs['axisc'] := ToPython(axisc);
    if (axis <> nil) then kwargs['axis'] := ToPython(axis^);
    py := FhModuleNumPy.InvokeMethod('cross', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNumPyArray.trapz(y: TNDarray ; x: TNDarray = nil; dx: Double = 1.0; axis: Integer = -1): Double;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([y]);
    kwargs := TPyDict.Create;
    if (x <> nil) then kwargs['x'] := ToPython(x);
    if (dx <> 1.0) then kwargs['dx'] := ToPython(dx);
    if (axis <> -1) then kwargs['axis'] := ToPython(axis);
    py := FhModuleNumPy.InvokeMethod('trapz', pyargs, kwargs);
    Result := TNumPy.ToCsharp<Double>(py);
end;

Function TNumPyArray.exp(x: TNDarray ; var _out : TNDarray; var where : TNDarray):TNDarray;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([x]);
    kwargs := TPyDict.Create;
    if (_out <> nil) then kwargs['out'] := ToPython(_out);
    if (where <> nil) then kwargs['where'] := ToPython(where);
    py := FhModuleNumPy.InvokeMethod('exp', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNumPyArray.expm1(x: TNDarray ; var _out : TNDarray; var where : TNDarray):TNDarray;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([x]);
    kwargs := TPyDict.Create;
    if (_out <> nil) then kwargs['out'] := ToPython(_out);
    if (where <> nil) then kwargs['where'] := ToPython(where);
    py := FhModuleNumPy.InvokeMethod('expm1', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNumPyArray.exp2(x: TNDarray ; var _out : TNDarray; var where : TNDarray):TNDarray;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([x]);
    kwargs := TPyDict.Create;
    if (_out <> nil) then kwargs['out'] := ToPython(_out);
    if (where <> nil) then kwargs['where'] := ToPython(where);
    py := FhModuleNumPy.InvokeMethod('exp2', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNumPyArray.log(x: TNDarray ; var _out : TNDarray; var where : TNDarray):TNDarray;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([x]);
    kwargs := TPyDict.Create;
    if (_out <> nil) then kwargs['out'] := ToPython(_out);
    if (where <> nil) then kwargs['where'] := ToPython(where);
    py := FhModuleNumPy.InvokeMethod('log', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNumPyArray.log10(x: TNDarray ; var _out : TNDarray; var where : TNDarray):TNDarray;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([x]);
    kwargs := TPyDict.Create;
    if (_out <> nil) then kwargs['out'] := ToPython(_out);
    if (where <> nil) then kwargs['where'] := ToPython(where);
    py := FhModuleNumPy.InvokeMethod('log10', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNumPyArray.log2(x: TNDarray ; var _out : TNDarray; var where : TNDarray):TNDarray;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([x]);
    kwargs := TPyDict.Create;
    if (_out <> nil) then kwargs['out'] := ToPython(_out);
    if (where <> nil) then kwargs['where'] := ToPython(where);
    py := FhModuleNumPy.InvokeMethod('log2', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNumPyArray.log1p(x: TNDarray ; var _out : TNDarray; var where : TNDarray):TNDarray;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([x]);
    kwargs := TPyDict.Create;
    if (_out <> nil) then kwargs['out'] := ToPython(_out);
    if (where <> nil) then kwargs['where'] := ToPython(where);
    py := FhModuleNumPy.InvokeMethod('log1p', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNumPyArray.logaddexp(x2: TNDarray ; x1: TNDarray ; var _out : TNDarray; var where : TNDarray):TNDarray;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([x2,x1]);
    kwargs := TPyDict.Create;
    if (_out <> nil) then kwargs['out'] := ToPython(_out);
    if (where <> nil) then kwargs['where'] := ToPython(where);
    py := FhModuleNumPy.InvokeMethod('logaddexp', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNumPyArray.logaddexp2(x2: TNDarray ; x1: TNDarray ; var _out : TNDarray; var where : TNDarray):TNDarray;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([x2,x1]);
    kwargs := TPyDict.Create;
    if (_out <> nil) then kwargs['out'] := ToPython(_out);
    if (where <> nil) then kwargs['where'] := ToPython(where);
    py := FhModuleNumPy.InvokeMethod('logaddexp2', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNumPyArray.sinc(x: TNDarray):TNDarray;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([x]);
    kwargs := TPyDict.Create;
    py := FhModuleNumPy.InvokeMethod('sinc', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNumPyArray.signbit(x: TNDarray ; var _out : TNDarray; var where : TNDarray):TNDarray;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([x]);
    kwargs := TPyDict.Create;
    if (_out <> nil) then kwargs['out'] := ToPython(_out);
    if (where <> nil) then kwargs['where'] := ToPython(where);
    py := FhModuleNumPy.InvokeMethod('signbit', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNumPyArray.copysign(x1: TNDarray ; x2: TNDarray ; var _out : TNDarray; var where : TNDarray):TNDarray;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([x1,x2]);
    kwargs := TPyDict.Create;
    if (_out <> nil) then kwargs['out'] := ToPython(_out);
    if (where <> nil) then kwargs['where'] := ToPython(where);
    py := FhModuleNumPy.InvokeMethod('copysign', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNumPyArray.frexp(x: TNDarray ; out1: TNDarray; out2: TNDarray ; var _out : TNDarray; var where : TNDarray): TArray<TNDarray>;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
   res    : TArray<TPythonObject>;
begin

    pyargs := TNumPy.ToTuple([x]);
    kwargs := TPyDict.Create;
    if (out1 <> nil) then kwargs['out1'] := ToPython(out1);
    if (out2 <> nil) then kwargs['out2'] := ToPython(out2);
    if (_out <> nil) then kwargs['out'] := ToPython(_out);
    if (where <> nil) then kwargs['where'] := ToPython(where);
    py := FhModuleNumPy.InvokeMethod('frexp', pyargs, kwargs);

    res := py.AsArrayofPyObj;

    SetLength(Result,2);
    Result[0] := TNumPy.ToCsharp<TNDarray>(res[0]);
    Result[1] := TNumPy.ToCsharp<TNDarray>(res[1]);
end;

Function TNumPyArray.ldexp(x1: TNDarray ; x2: TNDarray ; var _out : TNDarray; var where : TNDarray):TNDarray;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([x1,x2]);
    kwargs := TPyDict.Create;
    if (_out <> nil) then kwargs['out'] := ToPython(_out);
    if (where <> nil) then kwargs['where'] := ToPython(where);
    py := FhModuleNumPy.InvokeMethod('ldexp', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNumPyArray.nextafter(x1: TNDarray ; x2: TNDarray ; var _out : TNDarray; var where : TNDarray):TNDarray;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([x1,x2]);
    kwargs := TPyDict.Create;
    if (_out <> nil) then kwargs['out'] := ToPython(_out);
    if (where <> nil) then kwargs['where'] := ToPython(where);
    py := FhModuleNumPy.InvokeMethod('nextafter', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNumPyArray.spacing(x: TNDarray ; var _out : TNDarray; var where : TNDarray):TNDarray;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([x]);
    kwargs := TPyDict.Create;
    if (_out <> nil) then kwargs['out'] := ToPython(_out);
    if (where <> nil) then kwargs['where'] := ToPython(where);
    py := FhModuleNumPy.InvokeMethod('spacing', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNumPyArray.lcm(x2: TNDarray ; x1: TNDarray):TNDarray;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([x2,x1]);
    kwargs := TPyDict.Create;
    py := FhModuleNumPy.InvokeMethod('lcm', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNumPyArray.gcd(x2: TNDarray ; x1: TNDarray):TNDarray;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([x2,x1]);
    kwargs := TPyDict.Create;
    py := FhModuleNumPy.InvokeMethod('gcd', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNumPyArray.add(x2: TNDarray ; x1: TNDarray ; var _out : TNDarray; var where : TNDarray):TNDarray;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([x2,x1]);
    kwargs := TPyDict.Create;
    if (_out <> nil) then kwargs['out'] := ToPython(_out);
    if (where <> nil) then kwargs['where'] := ToPython(where);
    py := FhModuleNumPy.InvokeMethod('add', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNumPyArray.reciprocal(x: TNDarray ; var _out : TNDarray; var where : TNDarray):TNDarray;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([x]);
    kwargs := TPyDict.Create;
    if (_out <> nil) then kwargs['out'] := ToPython(_out);
    if (where <> nil) then kwargs['where'] := ToPython(where);
    py := FhModuleNumPy.InvokeMethod('reciprocal', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNumPyArray.positive(x: TNDarray):TNDarray;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([x]);
    kwargs := TPyDict.Create;
    py := FhModuleNumPy.InvokeMethod('positive', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNumPyArray.negative(x: TNDarray ; var _out : TNDarray; var where : TNDarray):TNDarray;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([x]);
    kwargs := TPyDict.Create;
    if (_out <> nil) then kwargs['out'] := ToPython(_out);
    if (where <> nil) then kwargs['where'] := ToPython(where);
    py := FhModuleNumPy.InvokeMethod('negative', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNumPyArray.multiply(x2: TNDarray ; x1: TNDarray ; var _out : TNDarray; var where : TNDarray):TNDarray;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([x2,x1]);
    kwargs := TPyDict.Create;
    if (_out <> nil) then kwargs['out'] := ToPython(_out);
    if (where <> nil) then kwargs['where'] := ToPython(where);
    py := FhModuleNumPy.InvokeMethod('multiply', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNumPyArray.divide(x1: TNDarray ; x2: TNDarray ; var _out : TNDarray; var where : TNDarray):TNDarray;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([x1,x2]);
    kwargs := TPyDict.Create;
    if (_out <> nil) then kwargs['out'] := ToPython(_out);
    if (where <> nil) then kwargs['where'] := ToPython(where);
    py := FhModuleNumPy.InvokeMethod('divide', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNumPyArray.power(x1: TNDarray ; x2: TNDarray ; var _out : TNDarray; var where : TNDarray):TNDarray;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([x1,x2]);
    kwargs := TPyDict.Create;
    if (_out <> nil) then kwargs['out'] := ToPython(_out);
    if (where <> nil) then kwargs['where'] := ToPython(where);
    py := FhModuleNumPy.InvokeMethod('power', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNumPyArray.subtract(x2: TNDarray ; x1: TNDarray ; var _out : TNDarray; var where : TNDarray):TNDarray;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([x2,x1]);
    kwargs := TPyDict.Create;
    if (_out <> nil) then kwargs['out'] := ToPython(_out);
    if (where <> nil) then kwargs['where'] := ToPython(where);
    py := FhModuleNumPy.InvokeMethod('subtract', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNumPyArray.true_divide(x1: TNDarray ; x2: TNDarray ; var _out : TNDarray; var where : TNDarray):TNDarray;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([x1,x2]);
    kwargs := TPyDict.Create;
    if (_out <> nil) then kwargs['out'] := ToPython(_out);
    if (where <> nil) then kwargs['where'] := ToPython(where);
    py := FhModuleNumPy.InvokeMethod('true_divide', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNumPyArray.floor_divide(x1: TNDarray ; x2: TNDarray ; var _out : TNDarray; var where : TNDarray):TNDarray;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([x1,x2]);
    kwargs := TPyDict.Create;
    if (_out <> nil) then kwargs['out'] := ToPython(_out);
    if (where <> nil) then kwargs['where'] := ToPython(where);
    py := FhModuleNumPy.InvokeMethod('floor_divide', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNumPyArray.float_power(x1: TNDarray ; x2: TNDarray ; var _out : TNDarray; var where : TNDarray):TNDarray;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([x1,x2]);
    kwargs := TPyDict.Create;
    if (_out <> nil) then kwargs['out'] := ToPython(_out);
    if (where <> nil) then kwargs['where'] := ToPython(where);
    py := FhModuleNumPy.InvokeMethod('float_power', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNumPyArray.fmod(x1: TNDarray ; x2: TNDarray ; var _out : TNDarray; var where : TNDarray):TNDarray;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([x1,x2]);
    kwargs := TPyDict.Create;
    if (_out <> nil) then kwargs['out'] := ToPython(_out);
    if (where <> nil) then kwargs['where'] := ToPython(where);
    py := FhModuleNumPy.InvokeMethod('fmod', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNumPyArray.&mod(x1: TNDarray ; x2: TNDarray ; var _out : TNDarray; var where : TNDarray):TNDarray;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([x1,x2]);
    kwargs := TPyDict.Create;
    if (_out <> nil) then kwargs['out'] := ToPython(_out);
    if (where <> nil) then kwargs['where'] := ToPython(where);
    py := FhModuleNumPy.InvokeMethod('mod', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNumPyArray.modf(x: TNDarray ; var _out : TNDarray; var where : TNDarray): TArray<TNDarray>;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
   res    : TArray<TPythonObject>;
begin

    pyargs := TNumPy.ToTuple([x]);
    kwargs := TPyDict.Create;
    if (_out <> nil) then kwargs['out'] := ToPython(_out);
    if (where <> nil) then kwargs['where'] := ToPython(where);
    py := FhModuleNumPy.InvokeMethod('modf', pyargs, kwargs);

    res := py.AsArrayofPyObj;

    SetLength(Result,2);
    Result[0] := TNumPy.ToCsharp<TNDarray>(res[0]);
    Result[1] := TNumPy.ToCsharp<TNDarray>(res[1]);

end;

Function TNumPyArray.remainder(x1: TNDarray ; x2: TNDarray ; var _out : TNDarray; var where : TNDarray):TNDarray;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([x1,x2]);
    kwargs := TPyDict.Create;
    if (_out <> nil) then kwargs['out'] := ToPython(_out);
    if (where <> nil) then kwargs['where'] := ToPython(where);
    py := FhModuleNumPy.InvokeMethod('remainder', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNumPyArray.divmod(x1: TNDarray ; x2: TNDarray ; var _out : TNDarray; var where : TNDarray): TArray<TNDarray>;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
   res    : TArray<TPythonObject>;
begin

    pyargs := TNumPy.ToTuple([x1,x2]);
    kwargs := TPyDict.Create;
    if (_out <> nil) then kwargs['out'] := ToPython(_out);
    if (where <> nil) then kwargs['where'] := ToPython(where);
    py := FhModuleNumPy.InvokeMethod('divmod', pyargs, kwargs);

    res := py.AsArrayofPyObj;

    SetLength(Result,2);
    Result[0] := TNumPy.ToCsharp<TNDarray>(res[0]);
    Result[1] := TNumPy.ToCsharp<TNDarray>(res[1]);
end;

Function TNumPyArray.angle(z: TNDarray ; deg : Boolean = false):TNDarray;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([z]);
    kwargs := TPyDict.Create;
    if (deg <> false) then kwargs['deg'] := ToPython(deg);
    py := FhModuleNumPy.InvokeMethod('angle', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNumPyArray.real(val: TNDarray):TNDarray;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([val]);
    kwargs := TPyDict.Create;
    py := FhModuleNumPy.InvokeMethod('real', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNumPyArray.imag(val: TNDarray):TNDarray;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([val]);
    kwargs := TPyDict.Create;
    py := FhModuleNumPy.InvokeMethod('imag', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNumPyArray.conj(x: TNDarray ; var _out : TNDarray; var where : TNDarray):TNDarray;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([x]);
    kwargs := TPyDict.Create;
    if (_out <> nil) then kwargs['out'] := ToPython(_out);
    if (where <> nil) then kwargs['where'] := ToPython(where);
    py := FhModuleNumPy.InvokeMethod('conj', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNumPyArray.convolve(a: TNDarray ; v: TNDarray ; mode: String = 'full'):TNDarray;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([a,v]);
    kwargs := TPyDict.Create;
    if (mode <> 'full') then kwargs['mode'] := ToPython(mode);
    py := FhModuleNumPy.InvokeMethod('convolve', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNumPyArray.clip(a: TNDarray ; a_min: TNDarray; a_max: TNDarray; var _out : TNDarray ):TNDarray;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([a,a_min,a_max]);
    kwargs := TPyDict.Create;
    if (_out <> nil) then kwargs['out'] := ToPython(_out);
    py := FhModuleNumPy.InvokeMethod('clip', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNumPyArray.sqrt(x: TNDarray ; var _out : TNDarray; var where : TNDarray):TNDarray;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([x]);
    kwargs := TPyDict.Create;
    if (_out <> nil) then kwargs['out'] := ToPython(_out);
    if (where <> nil) then kwargs['where'] := ToPython(where);
    py := FhModuleNumPy.InvokeMethod('sqrt', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNumPyArray.cbrt(x: TNDarray ; var _out : TNDarray; var where : TNDarray):TNDarray;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([x]);
    kwargs := TPyDict.Create;
    if (_out <> nil) then kwargs['out'] := ToPython(_out);
    if (where <> nil) then kwargs['where'] := ToPython(where);
    py := FhModuleNumPy.InvokeMethod('cbrt', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNumPyArray.abs(x: TNDarray ; var _out : TNDarray; var where : TNDarray):TNDarray;
begin
    Result := &absolute(x,_out,where);
end;

Function TNumPyArray.&absolute(x: TNDarray ; var _out : TNDarray; var where : TNDarray):TNDarray;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([x]);
    kwargs := TPyDict.Create;
    if (_out <> nil) then kwargs['out'] := ToPython(_out);
    if (where <> nil) then kwargs['where'] := ToPython(where);
    py := FhModuleNumPy.InvokeMethod('absolute', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNumPyArray.fabs(x: TNDarray ; var _out : TNDarray; var where : TNDarray):TNDarray;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([x]);
    kwargs := TPyDict.Create;
    if (_out <> nil) then kwargs['out'] := ToPython(_out);
    if (where <> nil) then kwargs['where'] := ToPython(where);
    py := FhModuleNumPy.InvokeMethod('fabs', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNumPyArray.sign(x: TNDarray ; var _out : TNDarray; var where : TNDarray):TNDarray;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([x]);
    kwargs := TPyDict.Create;
    if (_out <> nil) then kwargs['out'] := ToPython(_out);
    if (where <> nil) then kwargs['where'] := ToPython(where);
    py := FhModuleNumPy.InvokeMethod('sign', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNumPyArray.heaviside(x1: TNDarray ; x2: TNDarray ; var _out : TNDarray; var where : TNDarray):TNDarray;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([x1,x2]);
    kwargs := TPyDict.Create;
    if (_out <> nil) then kwargs['out'] := ToPython(_out);
    if (where <> nil) then kwargs['where'] := ToPython(where);
    py := FhModuleNumPy.InvokeMethod('heaviside', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNumPyArray.minimum(x2: TNDarray ; x1: TNDarray ; var _out : TNDarray; var where : TNDarray):TNDarray;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([x2,x1]);
    kwargs := TPyDict.Create;
    if (_out <> nil) then kwargs['out'] := ToPython(_out);
    if (where <> nil) then kwargs['where'] := ToPython(where);
    py := FhModuleNumPy.InvokeMethod('minimum', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNumPyArray.fmax(x2: TNDarray ; x1: TNDarray ; var _out : TNDarray; var where : TNDarray):TNDarray;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([x2,x1]);
    kwargs := TPyDict.Create;
    if (_out <> nil) then kwargs['out'] := ToPython(_out);
    if (where <> nil) then kwargs['where'] := ToPython(where);
    py := FhModuleNumPy.InvokeMethod('fmax', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNumPyArray.fmin(x2: TNDarray ; x1: TNDarray ; var _out : TNDarray; var where : TNDarray):TNDarray;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([x2,x1]);
    kwargs := TPyDict.Create;
    if (_out <> nil) then kwargs['out'] := ToPython(_out);
    if (where <> nil) then kwargs['where'] := ToPython(where);
    py := FhModuleNumPy.InvokeMethod('fmin', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNumPyArray.nan_to_num(x: TNDarray ; copy: Boolean = true):TNDarray;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([x]);
    kwargs := TPyDict.Create;
    if (copy <> true) then kwargs['copy'] := ToPython(copy);
    py := FhModuleNumPy.InvokeMethod('nan_to_num', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNumPyArray.real_if_close(a: TNDarray ; tol: double = 100):TNDarray;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([a]);
    kwargs := TPyDict.Create;
    if (tol <> 100) then kwargs['tol'] := ToPython(tol);
    py := FhModuleNumPy.InvokeMethod('real_if_close', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray>(py);
end;

(*
Function TNumPyArray. float or complex (corresponding to fp) or TNDarray interp(x: TNDarray ; 1-D sequence of xp: floats ; 1-D sequence of float or fp: complex ; optional float or complex corresponding to fp left = nil, optional float or complex corresponding to fp right = nil, None or float period = nil)
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([x,xp,fp]);
    kwargs := TPyDict.Create;
    if (left <> nil) then kwargs['left'] := ToPython(left);
    if (right <> nil) then kwargs['right'] := ToPython(right);
    if (period <> nil) then kwargs['period'] := ToPython(period);
    py := FhModuleNumPy.InvokeMethod('interp', pyargs, kwargs);
    Result := TNumPy.ToCsharp<float or complex (corresponding to fp) or ndarray>(py);
end;
*)

(*********************************** //END //np.math.gen.cs *************)
(************************************************************************)

(*********************************** //START //np.array_creation.gen.cs *)
(************************************************************************)
class function TNumPyArray.empty(shape : TArray<Integer>): TNDarray;
begin
    Result := empty(Tnp_Shape.Create(shape)) ;
end;

class function TNumPyArray.empty(shape: Tnp_Shape; dtype : TDtype = nil; order : PChar = nil): TNDarray;
var
  pyargs : TPyTuple;
  kwargs : TPyDict;
  py     : TPythonObject;
begin
    pyargs := TNumPy.ToTuple([TValue.FromShape(shape)]);
    kwargs := TPyDict.Create;

    if (dtype <> nil) then kwargs['dtype'] := TNumPy.ToPython(dtype);
    if (order <> nil) then kwargs['order'] := TNumPy.ToPython(order^);

    py := FhModuleNumPy.InvokeMethod('empty', pyargs, kwargs);

    Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNumPyArray.empty_like(prototype: TNDarray ; dtype: TDtype = nil; order: PChar = nil; subok  : Boolean =   true): TNDarray;
var
  pyargs : TPyTuple;
  kwargs : TPyDict;
  py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([prototype]);
    kwargs := TPyDict.Create;
    if (dtype<>nil) then  kwargs['dtype'] :=  TNumPy.ToPython(dtype);
    if (order<>nil) then  kwargs['order'] :=  TNumPy.ToPython(order);
    if (subok<>true) then  kwargs['subok'] :=  TNumPy.ToPython(subok);
    py := FhModuleNumPy.InvokeMethod('empty_like', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray>(py);
end;


Function TNumPyArray.empty_like<T>(prototype: TArray<T>; dtype  : TDtype  =  nil; order  : pChar  =  nil; subok  : Boolean =   true): TNDarray<T>;
var
  pyargs : TPyTuple;
  kwargs : TPyDict;
  py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([ ConvertArrayToNDarray<T>(prototype) ]);
    kwargs := TPyDict.Create;
    if (dtype<>nil) then  kwargs['dtype'] :=  TNumPy.ToPython(dtype);
    if (order<>nil) then  kwargs['order'] :=  TNumPy.ToPython(order);
    if (subok<>true) then  kwargs['subok'] :=  TNumPy.ToPython(subok);
    py := FhModuleNumPy.InvokeMethod('empty_like', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray<T>>(py);
end;


Function TNumPyArray.empty_like<T>(prototype: TArray2D<T>; dtype  : TDtype  =  nil; order  : pChar = nil; subok  : Boolean  =  true): TNDarray<T>;
var
  pyargs : TPyTuple;
  kwargs : TPyDict;
  py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([ ConvertArrayToNDarray<T>(prototype) ]);
    kwargs := TPyDict.Create;
    if (dtype<>nil) then  kwargs['dtype'] :=  TNumPy.ToPython(dtype);
    if (order<>nil) then  kwargs['order'] :=  TNumPy.ToPython(order);
    if (subok<>true) then  kwargs['subok'] :=  TNumPy.ToPython(subok);
    py := FhModuleNumPy.InvokeMethod('empty_like', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray<T>>(py);
end;


Function TNumPyArray.eye(N : Integer ; M  : PInteger  =  nil; k  : Integer  =  0; dtype  : TDtype  =  nil; order  : pChar = nil):TNDarray;
var
  pyargs : TPyTuple;
  kwargs : TPyDict;
  py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([N]);
    kwargs := TPyDict.Create;
    if (M<>nil) then  kwargs['M'] :=  TNumPy.ToPython(M^);
    if (k<>0) then  kwargs['k'] :=  TNumPy.ToPython(k);
    if (dtype<>nil) then  kwargs['dtype'] :=  TNumPy.ToPython(dtype);
    if (order<>nil) then  kwargs['order'] :=  TNumPy.ToPython(order);
    py := FhModuleNumPy.InvokeMethod('eye', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray>(py);
end;


Function TNumPyArray.identity(n : Integer ; dtype  : TDtype  =  nil):TNDarray;
var
  pyargs : TPyTuple;
  kwargs : TPyDict;
  py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([n]);
    kwargs := TPyDict.Create;
    if (dtype<>nil) then  kwargs['dtype'] :=  TNumPy.ToPython(dtype);
    py := FhModuleNumPy.InvokeMethod('identity', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray>(py);
end;

function TNumPyArray.ones(shape : TArray<Integer>): TNDarray;
begin
    Result := ones(Tnp_Shape.Create(shape)) ;
end;

Function TNumPyArray.ones(shape : Tnp_Shape ; dtype  : TDtype  =  nil; order  : pChar = nil):TNDarray;
var
  pyargs : TPyTuple;
  kwargs : TPyDict;
  py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([TValue.FromShape(shape)]);
    kwargs := TPyDict.Create;
    if (dtype<>nil) then  kwargs['dtype'] :=  TNumPy.ToPython(dtype);
    if (order<>nil) then  kwargs['order'] :=  TNumPy.ToPython(order);
    py := FhModuleNumPy.InvokeMethod('ones', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray>(py);
end;


Function TNumPyArray.ones_like(a : TNDarray ; dtype  : TDtype  =  nil; order  : pChar = nil; subok  : Boolean  =  true): TNDarray;
var
  pyargs : TPyTuple;
  kwargs : TPyDict;
  py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([a]);
    kwargs := TPyDict.Create;
    if (dtype<>nil) then  kwargs['dtype'] :=  TNumPy.ToPython(dtype);
    if (order<>nil) then  kwargs['order'] :=  TNumPy.ToPython(order);
    if (subok<>true) then  kwargs['subok'] :=  TNumPy.ToPython(subok);
    py := FhModuleNumPy.InvokeMethod('ones_like', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray>(py);
end;


Function TNumPyArray.ones_like<T>(a: TArray<T>; dtype  : TDtype  =  nil; order  : pChar = nil; subok  : Boolean  =  true): TNDarray<T>;
var
  pyargs : TPyTuple;
  kwargs : TPyDict;
  py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([ ConvertArrayToNDarray<T>(a)]);
    kwargs := TPyDict.Create;
    if (dtype<>nil) then  kwargs['dtype'] :=  TNumPy.ToPython(dtype);
    if (order<>nil) then  kwargs['order'] :=  TNumPy.ToPython(order);
    if (subok<>true) then  kwargs['subok'] :=  TNumPy.ToPython(subok);
    py := FhModuleNumPy.InvokeMethod('ones_like', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray<T>>(py);
end;


Function TNumPyArray.ones_like<T>(a: TArray2D<T>; dtype  : TDtype  =  nil; order  : pChar = nil; subok  : Boolean  =  true): TNDarray<T>;
var
  pyargs : TPyTuple;
  kwargs : TPyDict;
  py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([ ConvertArrayToNDarray<T>(a)]);
    kwargs := TPyDict.Create;
    if (dtype<>nil) then  kwargs['dtype'] :=  TNumPy.ToPython(dtype);
    if (order<>nil) then  kwargs['order'] :=  TNumPy.ToPython(order);
    if (subok<>true) then  kwargs['subok'] :=  TNumPy.ToPython(subok);
    py := FhModuleNumPy.InvokeMethod('ones_like', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray<T>>(py);
end;

class function TNumPyArray.zeros(shape : TArray<Integer>): TNDarray;
begin
    Result := zeros(Tnp_Shape.Create(shape)) ;
end;

class Function TNumPyArray.zeros(shape : Tnp_Shape ; dtype  : TDtype  =  nil; order  : pChar = nil): TNDarray;
var
  pyargs : TPyTuple;
  kwargs : TPyDict;
  py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([TValue.FromShape(shape)]);
    kwargs := TPyDict.Create;
    if (dtype<>nil) then  kwargs['dtype'] :=  TNumPy.ToPython(dtype);
    if (order<>nil) then  kwargs['order'] :=  TNumPy.ToPython(order);
    py := FhModuleNumPy.InvokeMethod('zeros', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray>(py);
end;


Function TNumPyArray.zeros_like(a : TNDarray ; dtype  : TDtype  =  nil; order  : pChar = nil; subok  : Boolean  =  true): TNDarray;
var
  pyargs : TPyTuple;
  kwargs : TPyDict;
  py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([a]);
    kwargs := TPyDict.Create;
    if (dtype<>nil) then  kwargs['dtype'] :=  TNumPy.ToPython(dtype);
    if (order<>nil) then  kwargs['order'] :=  TNumPy.ToPython(order);
    if (subok<>true) then  kwargs['subok'] :=  TNumPy.ToPython(subok);
    py := FhModuleNumPy.InvokeMethod('zeros_like', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray>(py);
end;


Function TNumPyArray.zeros_like<T>(a: TArray<T>; dtype  : TDtype  =  nil; order  : pChar = nil; subok  : Boolean  =  true): TNDarray<T>;
var
  pyargs : TPyTuple;
  kwargs : TPyDict;
  py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([ConvertArrayToNDarray<T>(a)]);
    kwargs := TPyDict.Create;
    if (dtype<>nil) then  kwargs['dtype'] :=  TNumPy.ToPython(dtype);
    if (order<>nil) then  kwargs['order'] :=  TNumPy.ToPython(order);
    if (subok<>true) then  kwargs['subok'] :=  TNumPy.ToPython(subok);
    py := FhModuleNumPy.InvokeMethod('zeros_like', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray<T>>(py);
end;


Function TNumPyArray.zeros_like<T>(a: TArray2D<T>; dtype  : TDtype  =  nil; order  : pChar = nil; subok  : Boolean  =  true): TNDarray<T>;
var
  pyargs : TPyTuple;
  kwargs : TPyDict;
  py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([ConvertArrayToNDarray<T>(a)]);
    kwargs := TPyDict.Create;
    if (dtype<>nil) then  kwargs['dtype'] :=  TNumPy.ToPython(dtype);
    if (order<>nil) then  kwargs['order'] :=  TNumPy.ToPython(order);
    if (subok<>true) then  kwargs['subok'] :=  TNumPy.ToPython(subok);
    py := FhModuleNumPy.InvokeMethod('zeros_like', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray<T>>(py);
end;


Function TNumPyArray.full(shape : Tnp_Shape ; fill_value: TValue; dtype  : TDtype  =  nil; order  : pChar = nil): TNDarray;
var
  pyargs : TPyTuple;
  kwargs : TPyDict;
  py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([ TValue.FromShape(shape), fill_value]);
    kwargs := TPyDict.Create;
    if (dtype<>nil) then  kwargs['dtype'] :=  TNumPy.ToPython(dtype);
    if (order<>nil) then  kwargs['order'] :=  TNumPy.ToPython(order);
    py := FhModuleNumPy.InvokeMethod('full', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray>(py);
end;


Function TNumPyArray.full_like(a : TNDarray ; fill_value: TValue; dtype  : TDtype  =  nil; order  : pChar = nil; subok  : Boolean  =  true): TNDarray;
var
  pyargs : TPyTuple;
  kwargs : TPyDict;
  py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([a,fill_value]);
    kwargs := TPyDict.Create;
    if (dtype<>nil) then  kwargs['dtype'] :=  TNumPy.ToPython(dtype);
    if (order<>nil) then  kwargs['order'] :=  TNumPy.ToPython(order);
    if (subok<>true) then  kwargs['subok'] :=  TNumPy.ToPython(subok);
    py := FhModuleNumPy.InvokeMethod('full_like', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray>(py);
end;


Function TNumPyArray.full_like<T>(a: TArray<T>; fill_value: TValue; dtype  : TDtype  =  nil; order  : pChar = nil; subok  : Boolean  =  true): TNDarray<T>;
var
  pyargs : TPyTuple;
  kwargs : TPyDict;
  py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([ConvertArrayToNDarray<T>(a), fill_value]);
    kwargs := TPyDict.Create;
    if (dtype<>nil) then  kwargs['dtype'] :=  TNumPy.ToPython(dtype);
    if (order<>nil) then  kwargs['order'] :=  TNumPy.ToPython(order);
    if (subok<>true) then  kwargs['subok'] :=  TNumPy.ToPython(subok);
    py := FhModuleNumPy.InvokeMethod('full_like', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray<T>>(py);
end;


Function TNumPyArray.full_like<T>(a:TArray2D<T>; fill_value: TValue; dtype  : TDtype  =  nil; order  : pChar = nil; subok  : Boolean  =  true): TNDarray<T>;
var
  pyargs : TPyTuple;
  kwargs : TPyDict;
  py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([ConvertArrayToNDarray<T>(a),fill_value]);
    kwargs := TPyDict.Create;
    if (dtype<>nil) then  kwargs['dtype'] :=  TNumPy.ToPython(dtype);
    if (order<>nil) then  kwargs['order'] :=  TNumPy.ToPython(order);
    if (subok<>true) then  kwargs['subok'] :=  TNumPy.ToPython(subok);
    py := FhModuleNumPy.InvokeMethod('full_like', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray<T>>(py);
end;


Function TNumPyArray.asarray(a : TNDarray ; dtype  : TDtype  =  nil; order  : pChar = nil): TNDarray;
var
  pyargs : TPyTuple;
  kwargs : TPyDict;
  py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([a]);
    kwargs := TPyDict.Create;
    if (dtype<>nil) then  kwargs['dtype'] :=  TNumPy.ToPython(dtype);
    if (order<>nil) then  kwargs['order'] :=  TNumPy.ToPython(order);
    py := FhModuleNumPy.InvokeMethod('asarray', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray>(py);
end;


Function TNumPyArray.asarray<T>(a: TArray<T>; dtype  : TDtype  =  nil; order  : pChar = nil): TNDarray<T>;
var
  pyargs : TPyTuple;
  kwargs : TPyDict;
  py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([ConvertArrayToNDarray<T>(a)]);
    kwargs := TPyDict.Create;
    if (dtype<>nil) then  kwargs['dtype'] :=  TNumPy.ToPython(dtype);
    if (order<>nil) then  kwargs['order'] :=  TNumPy.ToPython(order);
    py := FhModuleNumPy.InvokeMethod('asarray', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray<T>>(py);
end;


Function TNumPyArray.asarray<T>(a: TArray2D<T>; dtype  : TDtype  =  nil; order  : pChar = nil): TNDarray<T>;
var
  pyargs : TPyTuple;
  kwargs : TPyDict;
  py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([ConvertArrayToNDarray<T>(a)]);
    kwargs := TPyDict.Create;
    if (dtype<>nil) then  kwargs['dtype'] :=  TNumPy.ToPython(dtype);
    if (order<>nil) then  kwargs['order'] :=  TNumPy.ToPython(order);
    py := FhModuleNumPy.InvokeMethod('asarray', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray<T>>(py);
end;


Function TNumPyArray.asanyarray(a : TNDarray ; dtype  : TDtype  =  nil; order  : pChar = nil): TNDarray;
var
  pyargs : TPyTuple;
  kwargs : TPyDict;
  py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([a]);
    kwargs := TPyDict.Create;
    if (dtype<>nil) then  kwargs['dtype'] :=  TNumPy.ToPython(dtype);
    if (order<>nil) then  kwargs['order'] :=  TNumPy.ToPython(order);
    py := FhModuleNumPy.InvokeMethod('asanyarray', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray>(py);
end;


Function TNumPyArray.asanyarray<T>(a : TArray<T>; dtype  : TDtype  =  nil; order  : pChar = nil): TNDarray<T>;
var
  pyargs : TPyTuple;
  kwargs : TPyDict;
  py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([ConvertArrayToNDarray<T>(a)]);
    kwargs := TPyDict.Create;
    if (dtype<>nil) then  kwargs['dtype'] :=  TNumPy.ToPython(dtype);
    if (order<>nil) then  kwargs['order'] :=  TNumPy.ToPython(order);
    py := FhModuleNumPy.InvokeMethod('asanyarray', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray<T>>(py);
end;

Function TNumPyArray.asanyarray<T>(a : TArray2D<T>; dtype  : TDtype  =  nil; order  : pChar = nil): TNDarray<T>;
var
  pyargs : TPyTuple;
  kwargs : TPyDict;
  py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([ConvertArrayToNDarray<T>(a)]);
    kwargs := TPyDict.Create;
    if (dtype<>nil) then  kwargs['dtype'] :=  TNumPy.ToPython(dtype);
    if (order<>nil) then  kwargs['order'] :=  TNumPy.ToPython(order);
    py := FhModuleNumPy.InvokeMethod('asanyarray', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray<T>>(py);
end;

Function TNumPyArray.ascontiguousarray(a : TNDarray ; dtype  : TDtype  =  nil): TNDarray;
var
  pyargs : TPyTuple;
  kwargs : TPyDict;
  py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([a]);
    kwargs := TPyDict.Create;
    if (dtype<>nil) then  kwargs['dtype'] :=  TNumPy.ToPython(dtype);
    py := FhModuleNumPy.InvokeMethod('ascontiguousarray', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNumPyArray.ascontiguousarray<T>(a : TArray<T>; dtype  : TDtype  =  nil): TNDarray<T>;
var
  pyargs : TPyTuple;
  kwargs : TPyDict;
  py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([ConvertArrayToNDarray<T>(a)]);
    kwargs := TPyDict.Create;
    if (dtype<>nil) then  kwargs['dtype'] :=  TNumPy.ToPython(dtype);
    py := FhModuleNumPy.InvokeMethod('ascontiguousarray', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray<T>>(py);
end;

Function TNumPyArray.ascontiguousarray<T>(a : TArray2D<T>; dtype  : TDtype  =  nil): TNDarray<T>;
var
  pyargs : TPyTuple;
  kwargs : TPyDict;
  py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([ConvertArrayToNDarray<T>(a)]);
    kwargs := TPyDict.Create;
    if (dtype<>nil) then  kwargs['dtype'] :=  TNumPy.ToPython(dtype);
    py := FhModuleNumPy.InvokeMethod('ascontiguousarray', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray<T>>(py);
end;

Function TNumPyArray.asmatrix(data : TNDarray ; dtype: TDtype): TMatrix;
var
  pyargs : TPyTuple;
  kwargs : TPyDict;
  py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([data,
        dtype]);
    kwargs := TPyDict.Create;
    py := FhModuleNumPy.InvokeMethod('asmatrix', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TMatrix>(py);
end;

Function TNumPyArray.asmatrix<T>(data: TArray<T>; dtype: TDtype): TMatrix ;
var
  pyargs : TPyTuple;
  kwargs : TPyDict;
  py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([ConvertArrayToNDarray<T>(data),
        dtype]);
    kwargs := TPyDict.Create;
    py := FhModuleNumPy.InvokeMethod('asmatrix', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TMatrix>(py);
end;

Function TNumPyArray.asmatrix<T>(data: TArray2D<T>; dtype: TDtype): TMatrix ;
var
  pyargs : TPyTuple;
  kwargs : TPyDict;
  py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([ConvertArrayToNDarray<T>(data),
        dtype]);
    kwargs := TPyDict.Create;
    py := FhModuleNumPy.InvokeMethod('asmatrix', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TMatrix>(py);
end;

Function TNumPyArray.copy(a : TNDarray ; order  : pChar = nil): TNDarray;
var
  pyargs : TPyTuple;
  kwargs : TPyDict;
  py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([a]);
    kwargs := TPyDict.Create;
    if (order<>nil) then  kwargs['order'] :=  TNumPy.ToPython(order);
    py := FhModuleNumPy.InvokeMethod('copy', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNumPyArray.copy<T>(a : TArray<T>; order  : pChar = nil): TNDarray<T>;
var
  pyargs : TPyTuple;
  kwargs : TPyDict;
  py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([ConvertArrayToNDarray<T>(a)]);
    kwargs := TPyDict.Create;
    if (order<>nil) then  kwargs['order'] :=  TNumPy.ToPython(order);
    py := FhModuleNumPy.InvokeMethod('copy', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray<T>>(py);
end;

Function TNumPyArray.copy<T>(a : TArray2D<T>; order  : pChar = nil): TNDarray<T>;
var
  pyargs : TPyTuple;
  kwargs : TPyDict;
  py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([ConvertArrayToNDarray<T>(a)]);
    kwargs := TPyDict.Create;
    if (order<>nil) then  kwargs['order'] :=  TNumPy.ToPython(order);
    py := FhModuleNumPy.InvokeMethod('copy', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray<T>>(py);
end;

(*
Procedure frombuffer(buffer_like buffer, dtype  : TDtype  =  nil; PInteger count = -1, offset  : PInteger  =  0);
var
  pyargs : TPyTuple;
  kwargs : TPyDict;
  py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([buffer]);
    kwargs := TPyDict.Create;
    if (dtype<>nil) then  kwargs['dtype'] :=  TNumPy.ToPython(dtype);
    if (count<>-1) then  kwargs['count'] :=  TNumPy.ToPython(count);
    if (offset<>0) then  kwargs['offset'] :=  TNumPy.ToPython(offset);
    py := FhModuleNumPy.InvokeMethod('frombuffer', pyargs, kwargs);
end;
*)

Procedure TNumPyArray.fromfile(ffile : string ; dtype  : TDtype  =  nil; count : Integer = -1; sep : string = '');
var
  pyargs : TPyTuple;
  kwargs : TPyDict;

begin

    pyargs := TNumPy.ToTuple([ffile]);
    kwargs := TPyDict.Create;
    if (dtype<>nil) then  kwargs['dtype'] :=  TNumPy.ToPython(dtype);
    if (count<>-1) then  kwargs['count'] :=  TNumPy.ToPython(count);
    if (sep<>'') then  kwargs['sep'] :=  TNumPy.ToPython(sep);
    FhModuleNumPy.InvokeMethod('fromfile', pyargs, kwargs);
end;

(*
Function TNumPyArray.fromfunction(function : Delegate ; Shape shape, dtype  : TDtype  =  nil): PPyObject;
var
  pyargs : TPyTuple;
  kwargs : TPyDict;
  py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([function,
        shape]);
    kwargs := TPyDict.Create;
    if (dtype<>nil) then  kwargs['dtype'] :=  TNumPy.ToPython(dtype);
    py := FhModuleNumPy.InvokeMethod('fromfunction', pyargs, kwargs);
    Result := TNumPy.ToCsharp<PPyObject>(py);
end;

Function TNumPyArray.fromiter<T>(IEnumerable<T> iterable, TDtype dtype, PInteger count = -1): TNDarray<T>;
var
  pyargs : TPyTuple;
  kwargs : TPyDict;
  py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([iterable,
        dtype]);
    kwargs := TPyDict.Create;
    if (count<>-1) then  kwargs['count'] :=  TNumPy.ToPython(count);
    py := FhModuleNumPy.InvokeMethod('fromiter', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray<T>>(py);
end;
*)

Function TNumPyArray.fromstring(sStr: string; dtype  : TDtype  =  nil; count : Integer = -1; sep: string = ''): TNDarray;
var
  pyargs : TPyTuple;
  kwargs : TPyDict;
  py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([sStr]);
    kwargs := TPyDict.Create;
    if (dtype<>nil) then  kwargs['dtype'] :=  TNumPy.ToPython(dtype);
    if (count<>-1) then  kwargs['count'] :=  TNumPy.ToPython(count);
    if (sep<>'') then  kwargs['sep'] :=  TNumPy.ToPython(sep);
    py := FhModuleNumPy.InvokeMethod('fromstring', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray>(py);
end;

{ TODO regolarizzazione parametri -oMax -c :  16/02/2020 18:06:13 }
Function TNumPyArray.loadtxt(fname : string ; dtype  : TDtype  =  nil; comments : TArray<String> =  nil; delimiter  : pChar = nil; converters  : TArray<TVarRec>  =  nil; skiprows  : Integer  =  0; usecols : TArray<Integer>=  nil; unpack  : Boolean  =  false; ndmin  : Integer  =  0; encoding: string  = 'bytes'; max_rows: PInteger =  nil):TNDarray;
var
  pyargs : TPyTuple;
  kwargs : TPyDict;
  py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([fname]);
    kwargs := TPyDict.Create;
    if (dtype<>nil) then  kwargs['dtype'] :=  TNumPy.ToPython(dtype);
    if (comments<>nil) then  kwargs['comments'] :=  TNumPy.ToPython(TValue.FromArray<String>(comments));
    if (delimiter<>nil) then  kwargs['delimiter'] :=  TNumPy.ToPython(delimiter);
    if (converters<>nil) then  kwargs['converters'] :=  TNumPy.ToPython(TValue.FromArray<TVarRec>(converters));
    if (skiprows<>0) then  kwargs['skiprows'] :=  TNumPy.ToPython(skiprows);
    if (usecols<>nil) then  kwargs['usecols'] :=  TNumPy.ToPython(TValue.FromArray<Integer>(usecols));
    if (unpack<>false) then  kwargs['unpack'] :=  TNumPy.ToPython(unpack);
    if (ndmin<>0) then  kwargs['ndmin'] :=  TNumPy.ToPython(ndmin);
    if (encoding<>'bytes') then  kwargs['encoding'] :=  TNumPy.ToPython(encoding);
    if (max_rows<>nil) then  kwargs['max_rows'] :=  TNumPy.ToPython(max_rows^);
    py := FhModuleNumPy.InvokeMethod('loadtxt', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray>(py);
end;

(*
Function partial class core {
    Function partial class defchararray {
        Procedure array(string[] obj, itemsize  : PInteger  =  nil; copy  : PBoolean  =  true; unicode  : PBoolean  =  nil; order  : pChar = nil);
        {
            //auto-generated code, do not change
            var core = self.GetAttr('core');
            var defchararray = core.GetAttr('defchararray');
            var __self__=defchararray;
            pyargs := TNumPy.ToTuple(new object[]
            {
                obj,
            });
            kwargs := TPyDict.Create;
            if (itemsize<>nil) then  kwargs['itemsize'] :=  TNumPy.ToPython(itemsize);
            if (copy<>true) then  kwargs['copy'] :=  TNumPy.ToPython(copy);
            if (unicode<>nil) then  kwargs['unicode'] :=  TNumPy.ToPython(unicode);
            if (order<>nil) then  kwargs['order'] :=  TNumPy.ToPython(order);
            py := FhModuleNumPy.InvokeMethod('array', pyargs, kwargs);
        }
    }
end;
*)

Procedure TNumPyArray.chararray(shape : Tnp_Shape ; itemsize  : PInteger  =  nil; unicode  : PBoolean  =  nil; buffer  : PInteger  =  nil; offset  : PInteger  =  nil; strides: TArray<Integer> =  nil; order  : pChar = nil);
var
  pyargs : TPyTuple;
  kwargs : TPyDict;

begin

    pyargs := TNumPy.ToTuple([TValue.FromShape(shape)]);
    kwargs := TPyDict.Create;
    if (itemsize<>nil) then  kwargs['itemsize'] :=  TNumPy.ToPython(itemsize^);
    if (unicode<>nil) then  kwargs['unicode'] :=  TNumPy.ToPython(unicode^);
    if (buffer<>nil) then  kwargs['buffer'] :=  TNumPy.ToPython(buffer^);
    if (offset<>nil) then  kwargs['offset'] :=  TNumPy.ToPython(offset^);
    if (strides<>nil) then  kwargs['strides'] :=  TNumPy.ToPython( TValue.FromArray<Integer>(strides));
    if (order<>nil) then  kwargs['order'] :=  TNumPy.ToPython(order);
    FhModuleNumPy.InvokeMethod('chararray', pyargs, kwargs);
end;

(*
Function partial class core {
    Function partial class defchararray {
        Procedure asarray(string[] obj, itemsize  : PInteger  =  nil; unicode  : PBoolean  =  nil; order  : pChar = nil);
        {
            //auto-generated code, do not change
            var core = self.GetAttr('core');
            var defchararray = core.GetAttr('defchararray');
            var __self__=defchararray;
            pyargs := TNumPy.ToTuple(new object[]
            {
                obj,
            });
            kwargs := TPyDict.Create;
            if (itemsize<>nil) then  kwargs['itemsize'] :=  TNumPy.ToPython(itemsize);
            if (unicode<>nil) then  kwargs['unicode'] :=  TNumPy.ToPython(unicode);
            if (order<>nil) then  kwargs['order'] :=  TNumPy.ToPython(order);
            py := FhModuleNumPy.InvokeMethod('asarray', pyargs, kwargs);
        }
    }
end;
*)

class Function TNumPyArray.arange(start : byte ; stop: byte; step  : byte  =  1; dtype  : TDtype  =  nil): TNDarray;
var
  pyargs : TPyTuple;
  kwargs : TPyDict;
  py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([start,
        stop]);
    kwargs := TPyDict.Create;
    if (step<>1) then  kwargs['step'] :=  TNumPy.ToPython(step);
    if (dtype<>nil) then  kwargs['dtype'] :=  TNumPy.ToPython(dtype);
    py := FhModuleNumPy.InvokeMethod('arange', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray>(py);
end;

class Function TNumPyArray.arange(stop : byte ; step  : byte  =  1; dtype  : TDtype  =  nil): TNDarray;
var
  pyargs : TPyTuple;
  kwargs : TPyDict;
  py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([stop]);
    kwargs := TPyDict.Create;
    if (step<>1) then  kwargs['step'] :=  TNumPy.ToPython(step);
    if (dtype<>nil) then  kwargs['dtype'] :=  TNumPy.ToPython(dtype);
    py := FhModuleNumPy.InvokeMethod('arange', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray>(py);
end;

class Function TNumPyArray.arange(start : Word ; stop : Word; step  : Word  =  1; dtype  : TDtype  =  nil): TNDarray;
var
  pyargs : TPyTuple;
  kwargs : TPyDict;
  py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([start,
        stop]);
    kwargs := TPyDict.Create;
    if (step<>1) then  kwargs['step'] :=  TNumPy.ToPython(step);
    if (dtype<>nil) then  kwargs['dtype'] :=  TNumPy.ToPython(dtype);
    py := FhModuleNumPy.InvokeMethod('arange', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray>(py);
end;

class Function TNumPyArray.arange(stop : Word ; step  : Word  =  1; dtype  : TDtype  =  nil): TNDarray;
var
  pyargs : TPyTuple;
  kwargs : TPyDict;
  py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([stop]);
    kwargs := TPyDict.Create;
    if (step<>1) then  kwargs['step'] :=  TNumPy.ToPython(step);
    if (dtype<>nil) then  kwargs['dtype'] :=  TNumPy.ToPython(dtype);
    py := FhModuleNumPy.InvokeMethod('arange', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray>(py);
end;

class Function TNumPyArray.arange(start : Integer ; stop: Integer; step  : Integer  =  1; dtype  : TDtype  =  nil): TNDarray;
var
  pyargs : TPyTuple;
  kwargs : TPyDict;
  py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([start,
        stop]);
    kwargs := TPyDict.Create;
    if (step<>1) then  kwargs['step'] :=  TNumPy.ToPython(step);
    if (dtype<>nil) then  kwargs['dtype'] :=  TNumPy.ToPython(dtype);
    py := FhModuleNumPy.InvokeMethod('arange', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray>(py);
end;

class Function TNumPyArray.arange(stop : Integer ; step  : Integer  =  1; dtype  : TDtype  =  nil): TNDarray;
var
  pyargs : TPyTuple;
  kwargs : TPyDict;
  py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([stop]);
    kwargs := TPyDict.Create;
    if (step<>1) then  kwargs['step'] :=  TNumPy.ToPython(step);
    if (dtype<>nil) then  kwargs['dtype'] :=  TNumPy.ToPython(dtype);
    py := FhModuleNumPy.InvokeMethod('arange', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray>(py);
end;

class Function TNumPyArray.arange(start : Int64 ; stop: Int64; step  : Int64  =  1; dtype  : TDtype  =  nil): TNDarray;
var
  pyargs : TPyTuple;
  kwargs : TPyDict;
  py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([start,
        stop]);
    kwargs := TPyDict.Create;
    if (step<>1) then  kwargs['step'] :=  TNumPy.ToPython(step);
    if (dtype<>nil) then  kwargs['dtype'] :=  TNumPy.ToPython(dtype);
    py := FhModuleNumPy.InvokeMethod('arange', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray>(py);
end;

class Function TNumPyArray.arange(stop : int64 ; step  : Int64  =  1; dtype  : TDtype  =  nil): TNDarray;
var
  pyargs : TPyTuple;
  kwargs : TPyDict;
  py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([stop]);
    kwargs := TPyDict.Create;
    if (step<>1) then  kwargs['step'] :=  TNumPy.ToPython(step);
    if (dtype<>nil) then  kwargs['dtype'] :=  TNumPy.ToPython(dtype);
    py := FhModuleNumPy.InvokeMethod('arange', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray>(py);
end;

class Function TNumPyArray.arange(start : Single ; stop: Single; step  : Single  =  1; dtype  : TDtype  =  nil): TNDarray;
var
  pyargs : TPyTuple;
  kwargs : TPyDict;
  py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([start,
        stop]);
    kwargs := TPyDict.Create;
    if (step<>1) then  kwargs['step'] :=  TNumPy.ToPython(step);
    if (dtype<>nil) then  kwargs['dtype'] :=  TNumPy.ToPython(dtype);
    py := FhModuleNumPy.InvokeMethod('arange', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray>(py);
end;

class Function TNumPyArray.arange(stop : Single ; step  : Single  =  1; dtype  : TDtype  =  nil): TNDarray;
var
  pyargs : TPyTuple;
  kwargs : TPyDict;
  py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([stop]);
    kwargs := TPyDict.Create;
    if (step<>1) then  kwargs['step'] :=  TNumPy.ToPython(step);
    if (dtype<>nil) then  kwargs['dtype'] :=  TNumPy.ToPython(dtype);
    py := FhModuleNumPy.InvokeMethod('arange', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray>(py);
end;

class Function TNumPyArray.arange(start : double ; stop: double; step  : double  =  1; dtype  : TDtype  =  nil): TNDarray;
var
  pyargs : TPyTuple;
  kwargs : TPyDict;
  py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([start,
        stop]);
    kwargs := TPyDict.Create;
    if (step<>1) then  kwargs['step'] :=  TNumPy.ToPython(step);
    if (dtype<>nil) then  kwargs['dtype'] :=  TNumPy.ToPython(dtype);
    py := FhModuleNumPy.InvokeMethod('arange', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray>(py);
end;

class Function TNumPyArray.arange(stop : double ; step  : double  =  1; dtype  : TDtype  =  nil): TNDarray;
var
  pyargs : TPyTuple;
  kwargs : TPyDict;
  py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([stop]);
    kwargs := TPyDict.Create;
    if (step<>1) then  kwargs['step'] :=  TNumPy.ToPython(step);
    if (dtype<>nil) then  kwargs['dtype'] :=  TNumPy.ToPython(dtype);
    py := FhModuleNumPy.InvokeMethod('arange', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNumPyArray.linspace(start : TNDarray ; stop: TNDarray ; num  : Integer  =  50; endpoint  : Boolean  =  true; dtype  : TDtype  =  nil; axis  : Integer  =  0): TArray<TValue>;
var
  pyargs : TPyTuple;
  kwargs : TPyDict;
  py     : TPythonObject;
  res    : TArray<TPythonObject>;
begin

    pyargs := TNumPy.ToTuple([start, stop]);
    kwargs := TPyDict.Create;
    if (num<>50) then  kwargs['num'] :=  TNumPy.ToPython(num);
    if (endpoint<>true) then  kwargs['endpoint'] :=  TNumPy.ToPython(endpoint);
    if (dtype<>nil) then  kwargs['dtype'] :=  TNumPy.ToPython(dtype);
    if (axis<>0) then  kwargs['axis'] :=  TNumPy.ToPython(axis);
    py := FhModuleNumPy.InvokeMethod('linspace', pyargs, kwargs);

    res := py.AsArrayofPyObj;

    SetLength(Result,3);
    Result[0] := TNumPy.ToCsharp<TNDarray>(res[0]);
    Result[1] := TNumPy.ToCsharp<Double>(res[1]);

end;

Function TNumPyArray.linspace(start : double ; stop: double; num  : Integer  =  50; endpoint  : Boolean  =  true; dtype  : TDtype  =  nil; axis  : Integer  =  0): TNDarray;
var
  pyargs : TPyTuple;
  kwargs : TPyDict;
  py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([start, stop]);
    kwargs := TPyDict.Create;
    if (num<>50) then  kwargs['num'] :=  TNumPy.ToPython(num);
    if (endpoint<>true) then  kwargs['endpoint'] :=  TNumPy.ToPython(endpoint);
    if (dtype<>nil) then  kwargs['dtype'] :=  TNumPy.ToPython(dtype);
    if (axis<>0) then  kwargs['axis'] :=  TNumPy.ToPython(axis);
    py := FhModuleNumPy.InvokeMethod('linspace', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNumPyArray.logspace(start : TNDarray ; stop: TNDarray; num : Integer = 50; endpoint : Boolean = true; base : Double = 10.0; dtype  : TDtype  =  nil; axis  : Integer  =  0): TNDarray;
var
  pyargs : TPyTuple;
  kwargs : TPyDict;
  py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([start,
        stop]);
    kwargs := TPyDict.Create;
    if (num<>50) then  kwargs['num'] :=  TNumPy.ToPython(num);
    if (endpoint<>true) then  kwargs['endpoint'] :=  TNumPy.ToPython(endpoint);
    if (base<>10.0) then  kwargs['base'] :=  TNumPy.ToPython(@base);
    if (dtype<>nil) then  kwargs['dtype'] :=  TNumPy.ToPython(dtype);
    if (axis<>0) then  kwargs['axis'] :=  TNumPy.ToPython(axis);
    py := FhModuleNumPy.InvokeMethod('logspace', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNumPyArray.geomspace(start : TNDarray ; stop: TNDarray; num : Integer = 50; endpoint : Boolean  =  true; dtype  : TDtype  =  nil; axis  : Integer  =  0): TNDarray;
var
  pyargs : TPyTuple;
  kwargs : TPyDict;
  py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([start,
        stop]);
    kwargs := TPyDict.Create;
    if (num<>50) then  kwargs['num'] :=  TNumPy.ToPython(num);
    if (endpoint<>true) then  kwargs['endpoint'] :=  TNumPy.ToPython(endpoint);
    if (dtype<>nil) then  kwargs['dtype'] :=  TNumPy.ToPython(dtype);
    if (axis<>0) then  kwargs['axis'] :=  TNumPy.ToPython(axis);
    py := FhModuleNumPy.InvokeMethod('geomspace', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray>(py);
end;

(*
Function meshgrid(x2 : TNDarray ; TNDarray x1, indexing  : pChar = nil; sparse  : PBoolean  =  nil; copy  : PBoolean  =  nil): TNDarray;
var
  pyargs : TPyTuple;
  kwargs : TPyDict;
  py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([x2,
        x1]);
    kwargs := TPyDict.Create;
    if (indexing<>nil) then  kwargs['indexing'] :=  TNumPy.ToPython(indexing);
    if (sparse<>nil) then  kwargs['sparse'] :=  TNumPy.ToPython(sparse);
    if (copy<>nil) then  kwargs['copy'] :=  TNumPy.ToPython(copy);
    py := FhModuleNumPy.InvokeMethod('meshgrid', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray>(py);
end;
*)

(*
Function meshgrid<T>(T[] x2, array_like x1, indexing  : pChar = nil; sparse  : PBoolean  =  nil; copy  : PBoolean  =  nil):TNDarray<T>;
var
  pyargs : TPyTuple;
  kwargs : TPyDict;
  py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([ConvertArrayToNDarray<T>(x2), x1]);
    kwargs := TPyDict.Create;
    if (indexing<>nil) then  kwargs['indexing'] :=  TNumPy.ToPython(indexing);
    if (sparse<>nil) then  kwargs['sparse'] :=  TNumPy.ToPython(sparse);
    if (copy<>nil) then  kwargs['copy'] :=  TNumPy.ToPython(copy);
    py := FhModuleNumPy.InvokeMethod('meshgrid', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray<T>>(py);
end;
*)

(*
Function meshgrid<T>(T[,] x2, array_like x1, indexing  : pChar = nil; sparse  : PBoolean  =  nil; copy  : PBoolean  =  nil):TNDarray<T>;
var
  pyargs : TPyTuple;
  kwargs : TPyDict;
  py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([ConvertArrayToNDarray<T>(x2),
        x1]);
    kwargs := TPyDict.Create;
    if (indexing<>nil) then  kwargs['indexing'] :=  TNumPy.ToPython(indexing);
    if (sparse<>nil) then  kwargs['sparse'] :=  TNumPy.ToPython(sparse);
    if (copy<>nil) then  kwargs['copy'] :=  TNumPy.ToPython(copy);
    py := FhModuleNumPy.InvokeMethod('meshgrid', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray<T>>(py);
end;
*)

(*
Function meshgrid<T>(x2 : TNDarray ; T[] x1, indexing  : pChar = nil; sparse  : PBoolean  =  nil; copy  : PBoolean  =  nil):TNDarray<T>;
var
  pyargs : TPyTuple;
  kwargs : TPyDict;
  py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([x2,  ConvertArrayToNDarray<T>(x1)]);
    kwargs := TPyDict.Create;
    if (indexing<>nil) then  kwargs['indexing'] :=  TNumPy.ToPython(indexing);
    if (sparse<>nil) then  kwargs['sparse'] :=  TNumPy.ToPython(sparse);
    if (copy<>nil) then  kwargs['copy'] :=  TNumPy.ToPython(copy);
    py := FhModuleNumPy.InvokeMethod('meshgrid', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray<T>>(py);
end;
*)

(*
Function meshgrid<T>(x2 : TNDarray ; T[,] x1, indexing  : pChar = nil; sparse  : PBoolean  =  nil; copy  : PBoolean  =  nil):TNDarray<T>;
var
  pyargs : TPyTuple;
  kwargs : TPyDict;
  py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([x2, ConvertArrayToNDarray<T>(x1)]);
    kwargs := TPyDict.Create;
    if (indexing<>nil) then  kwargs['indexing'] :=  TNumPy.ToPython(indexing);
    if (sparse<>nil) then  kwargs['sparse'] :=  TNumPy.ToPython(sparse);
    if (copy<>nil) then  kwargs['copy'] :=  TNumPy.ToPython(copy);
    py := FhModuleNumPy.InvokeMethod('meshgrid', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray<T>>(py);
end;
*)

(*
Function meshgrid<T>(T[] x2, T[] x1, indexing  : pChar = nil; sparse  : PBoolean  =  nil; copy  : PBoolean  =  nil):TNDarray<T>;
var
  pyargs : TPyTuple;
  kwargs : TPyDict;
  py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([ConvertArrayToNDarray<T>(x2),  ConvertArrayToNDarray<T>(x1)]);
    kwargs := TPyDict.Create;
    if (indexing<>nil) then  kwargs['indexing'] :=  TNumPy.ToPython(indexing);
    if (sparse<>nil) then  kwargs['sparse'] :=  TNumPy.ToPython(sparse);
    if (copy<>nil) then  kwargs['copy'] :=  TNumPy.ToPython(copy);
    py := FhModuleNumPy.InvokeMethod('meshgrid', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray<T>>(py);
end;
*)

(*
Function meshgrid<T>(T[] x2, T[,] x1, indexing  : pChar = nil; sparse  : PBoolean  =  nil; copy  : PBoolean  =  nil):TNDarray<T>;
var
  pyargs : TPyTuple;
  kwargs : TPyDict;
  py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([ConvertArrayToNDarray<T>(x2),  ConvertArrayToNDarray<T>(x1)]);
    kwargs := TPyDict.Create;
    if (indexing<>nil) then  kwargs['indexing'] :=  TNumPy.ToPython(indexing);
    if (sparse<>nil) then  kwargs['sparse'] :=  TNumPy.ToPython(sparse);
    if (copy<>nil) then  kwargs['copy'] :=  TNumPy.ToPython(copy);
    py := FhModuleNumPy.InvokeMethod('meshgrid', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray<T>>(py);
end;
*)

(*
Function meshgrid<T>(T[,] x2, T[] x1, indexing  : pChar = nil; sparse  : PBoolean  =  nil; copy  : PBoolean  =  nil):TNDarray<T>;
var
  pyargs : TPyTuple;
  kwargs : TPyDict;
  py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([ConvertArrayToNDarray<T>(x2), ConvertArrayToNDarray<T>(x1)]);
    kwargs := TPyDict.Create;
    if (indexing<>nil) then  kwargs['indexing'] :=  TNumPy.ToPython(indexing);
    if (sparse<>nil) then  kwargs['sparse'] :=  TNumPy.ToPython(sparse);
    if (copy<>nil) then  kwargs['copy'] :=  TNumPy.ToPython(copy);
    py := FhModuleNumPy.InvokeMethod('meshgrid', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray<T>>(py);
end;
*)

(*
Function meshgrid<T>(T[,] x2, T[,] x1, indexing  : pChar = nil; sparse  : PBoolean  =  nil; copy  : PBoolean  =  nil):TNDarray<T>;
var
  pyargs : TPyTuple;
  kwargs : TPyDict;
  py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([ConvertArrayToNDarray<T>(x2), ConvertArrayToNDarray<T>(x1)]);
    kwargs := TPyDict.Create;
    if (indexing<>nil) then  kwargs['indexing'] :=  TNumPy.ToPython(indexing);
    if (sparse<>nil) then  kwargs['sparse'] :=  TNumPy.ToPython(sparse);
    if (copy<>nil) then  kwargs['copy'] :=  TNumPy.ToPython(copy);
    py := FhModuleNumPy.InvokeMethod('meshgrid', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray<T>>(py);
end;
*)

Procedure TNumPyArray.mgrid;
var
  py     : TPythonObject;
begin

    py := FhModuleNumPy.InvokeMethod('mgrid');
end;

(*
Procedure ogrid(math mesh-grid `ndarrays` with only one dimension)
var
  pyargs : TPyTuple;
  kwargs : TPyDict;
  py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([mesh-grid `ndarrays` with only one dimension]);
    kwargs := TPyDict.Create;
    py := FhModuleNumPy.InvokeMethod('ogrid', pyargs, kwargs);
end;
*)

Function TNumPyArray.diag(v : TNDarray ; k  : Integer  =  0):TNDarray;
var
  pyargs : TPyTuple;
  kwargs : TPyDict;
  py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([v]);
    kwargs := TPyDict.Create;
    if (k<>0) then  kwargs['k'] :=  TNumPy.ToPython(k);
    py := FhModuleNumPy.InvokeMethod('diag', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNumPyArray.diag<T>(v: TArray<T>; k  : Integer  =  0):TNDarray<T>;
var
  pyargs : TPyTuple;
  kwargs : TPyDict;
  py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([ConvertArrayToNDarray<T>(v)]);
    kwargs := TPyDict.Create;
    if (k<>0) then  kwargs['k'] :=  TNumPy.ToPython(k);
    py := FhModuleNumPy.InvokeMethod('diag', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray<T>>(py);
end;

Function TNumPyArray.diag<T>(v: TArray2D<T>; k  : Integer  =  0): TNDarray<T>;
var
  pyargs : TPyTuple;
  kwargs : TPyDict;
  py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([ConvertArrayToNDarray<T>(v)]);
    kwargs := TPyDict.Create;
    if (k<>0) then  kwargs['k'] :=  TNumPy.ToPython(k);
    py := FhModuleNumPy.InvokeMethod('diag', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray<T>>(py);
end;

Function TNumPyArray.diagflat(v : TNDarray ; k  : Integer  =  0):TNDarray;
var
  pyargs : TPyTuple;
  kwargs : TPyDict;
  py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([v]);
    kwargs := TPyDict.Create;
    if (k<>0) then  kwargs['k'] :=  TNumPy.ToPython(k);
    py := FhModuleNumPy.InvokeMethod('diagflat', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNumPyArray.diagflat<T>(v: TArray<T>; k  : Integer  =  0):TNDarray<T>;
var
  pyargs : TPyTuple;
  kwargs : TPyDict;
  py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([ConvertArrayToNDarray<T>(v)]);
    kwargs := TPyDict.Create;
    if (k<>0) then  kwargs['k'] :=  TNumPy.ToPython(k);
    py := FhModuleNumPy.InvokeMethod('diagflat', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray<T>>(py);
end;

Function TNumPyArray.diagflat<T>(v: TArray2D<T>; k  : Integer  =  0):TNDarray<T>;
var
  pyargs : TPyTuple;
  kwargs : TPyDict;
  py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([ConvertArrayToNDarray<T>(v)]);
    kwargs := TPyDict.Create;
    if (k<>0) then  kwargs['k'] :=  TNumPy.ToPython(k);
    py := FhModuleNumPy.InvokeMethod('diagflat', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray<T>>(py);
end;

Function TNumPyArray.tri(N : Integer ; M  : PInteger  =  nil; k  : Integer  =  0; dtype  : TDtype  =  nil):TNDarray;
var
  pyargs : TPyTuple;
  kwargs : TPyDict;
  py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([N]);
    kwargs := TPyDict.Create;
    if (M<>nil) then  kwargs['M'] :=  TNumPy.ToPython(M^);
    if (k<>0) then  kwargs['k'] :=  TNumPy.ToPython(k);
    if (dtype<>nil) then  kwargs['dtype'] :=  TNumPy.ToPython(dtype);
    py := FhModuleNumPy.InvokeMethod('tri', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNumPyArray.tril(m : TNDarray ; k  : Integer  =  0):TNDarray;
var
  pyargs : TPyTuple;
  kwargs : TPyDict;
  py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([m]);
    kwargs := TPyDict.Create;
    if (k<>0) then  kwargs['k'] :=  TNumPy.ToPython(k);
    py := FhModuleNumPy.InvokeMethod('tril', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNumPyArray.tril<T>(m: TArray<T>; k  : Integer  =  0):TNDarray<T>;
var
  pyargs : TPyTuple;
  kwargs : TPyDict;
  py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([ConvertArrayToNDarray<T>(m)]);
    kwargs := TPyDict.Create;
    if (k<>0) then  kwargs['k'] :=  TNumPy.ToPython(k);
    py := FhModuleNumPy.InvokeMethod('tril', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray<T>>(py);
end;

Function TNumPyArray.tril<T>(m: TArray2D<T>; k  : Integer  =  0):TNDarray<T>;
var
  pyargs : TPyTuple;
  kwargs : TPyDict;
  py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([ConvertArrayToNDarray<T>(m)]);
    kwargs := TPyDict.Create;
    if (k<>0) then  kwargs['k'] :=  TNumPy.ToPython(k);
    py := FhModuleNumPy.InvokeMethod('tril', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray<T>>(py);
end;

Function TNumPyArray.vander(x : TNDarray ; N  : PInteger  =  nil; increasing  : Boolean  =  false):TNDarray;
var
  pyargs : TPyTuple;
  kwargs : TPyDict;
  py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([x]);
    kwargs := TPyDict.Create;
    if (N<>nil) then  kwargs['N'] :=  TNumPy.ToPython(N^);
    if (increasing<>false) then  kwargs['increasing'] :=  TNumPy.ToPython(increasing);
    py := FhModuleNumPy.InvokeMethod('vander', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray>(py);
end;


Function TNumPyArray.vander<T>(x: TArray<T>; N  : PInteger  =  nil; increasing  : Boolean  =  false):TNDarray<T>;
var
  pyargs : TPyTuple;
  kwargs : TPyDict;
  py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([ConvertArrayToNDarray<T>(x)]);
    kwargs := TPyDict.Create;
    if (N<>nil) then  kwargs['N'] :=  TNumPy.ToPython(N^);
    if (increasing<>false) then  kwargs['increasing'] :=  TNumPy.ToPython(increasing);
    py := FhModuleNumPy.InvokeMethod('vander', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray<T>>(py);
end;


Function TNumPyArray.vander<T>(x: TArray2D<T>; N  : PInteger  =  nil; increasing  : Boolean  =  false):TNDarray<T>;
var
  pyargs : TPyTuple;
  kwargs : TPyDict;
  py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([ConvertArrayToNDarray<T>(x)]);
    kwargs := TPyDict.Create;
    if (N<>nil) then  kwargs['N'] :=  TNumPy.ToPython(N^);
    if (increasing<>false) then  kwargs['increasing'] :=  TNumPy.ToPython(increasing);
    py := FhModuleNumPy.InvokeMethod('vander', pyargs, kwargs);
    Result := TNumPy.ToCsharp<TNDarray<T>>(py);
end;

(*
Function mat(data : TNDarray ; TDtype dtype):TMatrix;
var
  pyargs : TPyTuple;
  kwargs : TPyDict;
  py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([data,
        dtype]);
    kwargs := TPyDict.Create;
    py := FhModuleNumPy.InvokeMethod('mat', pyargs, kwargs);
    Result := TNumPy.ToCsharp<Matrix>(py);
end;
*)

(*
Function mat<T>(T[] data, TDtype dtype): TMatrix;
var
  pyargs : TPyTuple;
  kwargs : TPyDict;
  py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([ConvertArrayToNDarray<T>(data),
        dtype]);
    kwargs := TPyDict.Create;
    py := FhModuleNumPy.InvokeMethod('mat', pyargs, kwargs);
    Result := TNumPy.ToCsharp<Matrix>(py);
end;
*)

(*
Function mat<T>(T[,] data, TDtype dtype): TMatrix;
var
  pyargs : TPyTuple;
  kwargs : TPyDict;
  py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([ConvertArrayToNDarray<T>(data),
        dtype]);
    kwargs := TPyDict.Create;
    py := FhModuleNumPy.InvokeMethod('mat', pyargs, kwargs);
    Result := TNumPy.ToCsharp<Matrix>(py);
end;
*)

(*
Function bmat(obj : string ; ldict  : Hashtable  =  nil; gdict  : Hashtable  =  nil): TMatrix;
var
  pyargs : TPyTuple;
  kwargs : TPyDict;
  py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([obj]);
    kwargs := TPyDict.Create;
    if (ldict<>nil) then  kwargs['ldict'] :=  TNumPy.ToPython(ldict);
    if (gdict<>nil) then  kwargs['gdict'] :=  TNumPy.ToPython(gdict);
    py := FhModuleNumPy.InvokeMethod('bmat', pyargs, kwargs);
    Result := TNumPy.ToCsharp<Matrix>(py);
end;
*)

(*
Function bmat<T>(T[] obj, ldict  : Hashtable  =  nil; gdict  : Hashtable  =  nil):Matrix<T>;
var
  pyargs : TPyTuple;
  kwargs : TPyDict;
  py     : TPythonObject;
begin

    pyargs := TNumPy.ToTuple([ConvertArrayToNDarray<T>(obj)]);
    kwargs := TPyDict.Create;
    if (ldict<>nil) then  kwargs['ldict'] :=  TNumPy.ToPython(ldict);
    if (gdict<>nil) then  kwargs['gdict'] :=  TNumPy.ToPython(gdict);
    py := FhModuleNumPy.InvokeMethod('bmat', pyargs, kwargs);
    Result := TNumPy.ToCsharp<Matrix<T>>(py);
end;
*)

(*********************************** //END //np.array_creation.gen.cs ***)
(************************************************************************)

(*********************************** //START //np.array.cs **************)
(************************************************************************)

class function TNumPyArray.npArray(obj: TNDarray; dtype: TDtype; copy: PBoolean; order: PChar; subok: PBoolean; ndmin: PInteger): TNDarray;
var
  pyargs : TPyTuple;
  kwargs : TPyDict;
  py     : TPythonObject;
begin
    pyargs := TNumPy.ToTuple([obj]);
    kwargs := TPyDict.Create;


    if (dtype <> nil) then kwargs['dtype'] := TNumPy.ToPython(dtype);
    if (copy <> nil)  then kwargs['copy']  := TNumPy.ToPython(copy^);
    if (order <> nil) then kwargs['order'] := TNumPy.ToPython(order^);
    if (subok <> nil) then kwargs['subok'] := TNumPy.ToPython(subok^);
    if (ndmin <> nil) then kwargs['ndmin'] := TNumPy.ToPython(ndmin^);

    py := FhModuleNumPy.InvokeMethod('array', pyargs, kwargs);

    Result := TNumPy.ToCsharp<TNDarray>(py);

end;

class function TNumPyArray.npArray(obj: TArray<TNDarray>; dtype: TDtype; copy: PBoolean; order: PChar; subok: PBoolean;  ndmin: PInteger): TNDarray;
var
  pydarray : TPythonObject;

  pyargs : TPyTuple;
  kwargs : TPyDict;
begin

   pyargs := TNumPy.ToTuple([TValue.FromArray<TNDarray>(obj)]);
   kwargs := TPyDict.Create;

   if (dtype <> nil) then kwargs['dtype'] := TNumPy.ToPython(dtype);
   if (copy <> nil)  then kwargs['copy']  := TNumPy.ToPython(copy^);
   if (order <> nil) then kwargs['order'] := TNumPy.ToPython(order^);
   if (subok <> nil) then kwargs['subok'] := TNumPy.ToPython(subok^);
   if (ndmin <> nil) then kwargs['ndmin'] := TNumPy.ToPython(ndmin^);

   pydarray := FhModuleNumPy.InvokeMethod('array', pyargs, kwargs);

   Result := TNumPy.ToCsharp< TNDarray >(pydarray);

end;

class function TNumPyArray.npArray(obj: TArray<String>; itemsize : PInteger = nil; copy: PBoolean = nil; unicode: PBoolean = nil ;order: PChar= nil): TNDarray;
var
  pydarray : TPythonObject;

  pyargs : TPyTuple;
  kwargs : TPyDict;
begin


   pyargs := TNumPy.ToTuple([ TValue.FromArray<string>(obj) ]);
   kwargs := TPyDict.Create;

   if (itemsize <> nil)  then kwargs['itemsize']:= TNumPy.ToPython(itemsize^);
   if (copy <> nil)      then kwargs['copy']    := TNumPy.ToPython(copy^);
   if (unicode <> nil)   then kwargs['unicode'] := TNumPy.ToPython(unicode^);
   if (order <> nil)    then kwargs['order']    := TNumPy.ToPython(order^);

   pydarray := FhModuleNumPy.InvokeMethod('array', pyargs, kwargs);

   Result := TNumPy.ToCsharp< TNDarray >(pydarray);

end;

class function TNumPyArray.npArray<T>(obj: TArray<T>; dtype: TDtype; copy: PBoolean; order: PChar; subok: PBoolean;  ndmin: PInteger): TNDarray<T>;
var
  Tipo     : TDtype;
  ndarray  : TNDarray;
  parArr   : TArray<Integer>;
  varArray : Variant;
  I        : Integer;
  py       : PPyObject;
  pydarray : TPythonObject;

  pyargs : TPyTuple;
  kwargs : TPyDict;
begin
   tipo := TDtypeExtensions.GetDtype< TArray<T> >(obj) ;

   //simulate Param c#
   parArr := parArr + [ Length(obj)  ] ;

   ndarray := empty( tnp_Shape.Create(parArr), tipo, order);

   if Length(obj) < 1 then
     exit( TNDarray<T>.Create(ndarray) ) ;

   varArray := VarArrayCreate([0,Length(obj) -1], varVariant );

   for I := VarArrayLowBound(varArray, 1) to VarArrayHighBound(varArray, 1) do
   begin
      { Put the element I at index I. }
      VarArrayPut(varArray, TValue.From<T>(obj[I]).AsVariant , [I]);
   end;

   py := g_MyPyEngine.VariantAsPyObject(varArray) ;

   pyargs := TNumPy.ToTuple([TValue.From<PPyObject>(py)]);
   kwargs := TPyDict.Create;

   if (dtype <> nil) then kwargs['dtype'] := TNumPy.ToPython(dtype);
   if (copy <> nil)  then kwargs['copy']  := TNumPy.ToPython(copy^);
   if (order <> nil) then kwargs['order'] := TNumPy.ToPython(order^);
   if (subok <> nil) then kwargs['subok'] := TNumPy.ToPython(subok^);
   if (ndmin <> nil) then kwargs['ndmin'] := TNumPy.ToPython(ndmin^);

   pydarray := FhModuleNumPy.InvokeMethod('array', pyargs, kwargs);

   Result := TNumPy.ToCsharp< TNDarray<T> >(pydarray);

end;

class function TNumPyArray.npArray<T>(obj: TArray2D<T>; dtype: TDtype; copy: PBoolean; order: PChar; subok: PBoolean;  ndmin: PInteger): TNDarray<T>;
var
  Tipo   : TDtype;
  ndarray: TNDarray;
  parArr : TArray<Integer>;
  varArray : Variant;
  I,Y : Integer;
  py : PPyObject;
  pydarray : TPythonObject;

  pyargs : TPyTuple;
  kwargs : TPyDict;
begin
   tipo := TDtypeExtensions.GetDtype< TArray<T> >(obj[0]) ;

   //simulate Param c#
   parArr := parArr + [ Length(obj)  ] ;
   parArr := parArr + [ Length(obj[0])  ] ;

   ndarray := empty( tnp_Shape.Create(parArr), tipo, order);

   if Length(obj) < 1 then
     exit( TNDarray<T>.Create(ndarray) ) ;

   varArray := VarArrayCreate([0,Length(obj) -1,0,Length(obj[0]) -1], varVariant );

   for I := VarArrayLowBound(varArray, 1) to VarArrayHighBound(varArray, 1) do
   begin
      for Y := VarArrayLowBound(varArray, 2) to VarArrayHighBound(varArray, 2) do
      begin
          { Put the element I at index I. }
          VarArrayPut(varArray, TValue.From<T>(obj[I][Y]).AsVariant , [I,Y]);
      end;
   end;

   py := g_MyPyEngine.VariantAsPyObject(varArray) ;

   pyargs := TNumPy.ToTuple([TValue.From<PPyObject>(py)]);
   kwargs := TPyDict.Create;

   if (dtype <> nil) then kwargs['dtype'] := TNumPy.ToPython(dtype);
   if (copy <> nil)  then kwargs['copy']  := TNumPy.ToPython(copy^);
   if (order <> nil) then kwargs['order'] := TNumPy.ToPython(order^);
   if (subok <> nil) then kwargs['subok'] := TNumPy.ToPython(subok^);
   if (ndmin <> nil) then kwargs['ndmin'] := TNumPy.ToPython(ndmin^);

   pydarray := FhModuleNumPy.InvokeMethod('array', pyargs, kwargs);

   Result := TNumPy.ToCsharp< TNDarray<T> >(pydarray);

end;

class function TNumPyArray.npArray<T>(obj: TArray3D<T>; dtype: TDtype; copy: PBoolean; order: PChar; subok: PBoolean;  ndmin: PInteger): TNDarray<T>;
var
  Tipo   : TDtype;
  ndarray: TNDarray;
  parArr : TArray<Integer>;
  varArray : Variant;
  I,Y,Z : Integer;
  py : PPyObject;
  pydarray : TPythonObject;

  pyargs : TPyTuple;
  kwargs : TPyDict;
begin
   tipo := TDtypeExtensions.GetDtype< TArray<T> >(obj[0][0]) ;

   //simulate Param c#
   parArr := parArr + [ Length(obj)  ] ;
   parArr := parArr + [ Length(obj[0])  ] ;
   parArr := parArr + [ Length(obj[0][0])  ] ;

   ndarray := empty( tnp_Shape.Create(parArr), tipo, order);

   if Length(obj) < 1 then
     exit( TNDarray<T>.Create(ndarray) ) ;

   varArray := VarArrayCreate([0,Length(obj) -1,0,Length(obj[0]) -1,0,Length(obj[0][0]) -1], varVariant );

   for I := VarArrayLowBound(varArray, 1) to VarArrayHighBound(varArray, 1) do
   begin
       for Y := VarArrayLowBound(varArray, 2) to VarArrayHighBound(varArray, 2) do
       begin
          for Z := VarArrayLowBound(varArray, 3) to VarArrayHighBound(varArray, 3) do
          begin
              { Put the element I at index I. }
              VarArrayPut(varArray, TValue.From<T>(obj[I][Y][Z]).AsVariant , [I,Y,Z]);
          end;
       end;
   end;

   py := g_MyPyEngine.VariantAsPyObject(varArray) ;

   pyargs := TNumPy.ToTuple([TValue.From<PPyObject>(py)]);
   kwargs := TPyDict.Create;

   if (dtype <> nil) then kwargs['dtype'] := TNumPy.ToPython(dtype);
   if (copy <> nil)  then kwargs['copy']  := TNumPy.ToPython(copy^);
   if (order <> nil) then kwargs['order'] := TNumPy.ToPython(order^);
   if (subok <> nil) then kwargs['subok'] := TNumPy.ToPython(subok^);
   if (ndmin <> nil) then kwargs['ndmin'] := TNumPy.ToPython(ndmin^);

   pydarray := FhModuleNumPy.InvokeMethod('array', pyargs, kwargs);

   Result := TNumPy.ToCsharp< TNDarray<T> >(pydarray);

end;

class Function  TNumPyArray.asarray(scalar : TValue; dtype: TDtype): TNDarray;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple([scalar]);
  kwargs := TPyDict.Create;
  if (dtype<>nil) then kwargs['dtype'] := TNumPy.ToPython(dtype);

  py := FhModuleNumPy.InvokeMethod('asarray', pyargs, kwargs);

  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

class function TNumPyArray.asscalar<T>(a: TNDarray): T;
var
  py : TPythonObject;
begin
    py := FhModuleNumPy.InvokeMethod('asscalar', [a]);

    Result := TNumPy.ToCsharp<T>(py);

end;
(*********************************** //END //np.array.cs ****************)
(************************************************************************)


(*********************************** //START //np.random.cs *************)
(************************************************************************)
Function  TNumPyArray.rand(shape: TArray<Integer>):TNDarray;
var
  random : TPythonObject;
  pyargs : TPyTuple;
  kwargs : TPyDict;
  py     : TPythonObject;
begin
    //auto-generated code, do not change
    random := FhModuleNumPy.GetAttr('random');
    //var __self__ = random;
    pyargs := ToTuple(TValue.ArrayOfToValueArray<Integer>(shape));
    kwargs := TPyDict.Create;

    py := random.InvokeMethod('rand', pyargs, kwargs);
    Result :=  TNumPy.ToCsharp<TNDarray>(py);
end;

Function  TNumPyArray.randn(shape: TArray<Integer>):TNDarray;
var
  random : TPythonObject;
  pyargs : TPyTuple;
  kwargs : TPyDict;
  py     : TPythonObject;
begin
    //auto-generated code, do not change
    random := FhModuleNumPy.GetAttr('random');
    //var __self__ = random;
    pyargs := ToTuple(TValue.ArrayOfToValueArray<Integer>(shape));
    kwargs := TPyDict.Create;

    py := random.InvokeMethod('randn', pyargs, kwargs);
    Result :=  TNumPy.ToCsharp<TNDarray>(py);

end;
(*********************************** //END //np.random.cs *************)
(************************************************************************)

(*********************************** //START //np.linalg.norm.cs *************)
(************************************************************************)

function TNumPyArray.norm(x: TNDarray; ord: PInteger; axis: TArray<Integer>; keepdims: PBoolean = nil): TNDarray;
var
  linalg : TPythonObject;
  pyargs : TPyTuple;
  kwargs : TPyDict;
  py     : TPythonObject;
begin
   pyargs := TNumPy.ToTuple([x]);
   kwargs := TPyDict.Create;

   if (ord <> nil)      then kwargs['ord'] := TNumPy.ToPython(ord^);
   if (axis <> nil)     then kwargs['axis']  := TNumPy.ToPython( TValue.FromArray<Integer>(axis));
   if (keepdims <> nil) then kwargs['keepdims'] := TNumPy.ToPython(keepdims^);

   linalg := FhModuleNumPy.GetAttr('linalg');
   py := linalg.InvokeMethod('norm', pyargs, kwargs);

   Result := TNumPy.ToCsharp<TNDarray>(py);
end;

function TNumPyArray.norm(x: TNDarray; ord: PInteger = nil): Double;
var
  linalg : TPythonObject;
  pyargs : TPyTuple;
  kwargs : TPyDict;
  py     : TPythonObject;
begin
   pyargs := TNumPy.ToTuple([x]);
   kwargs := TPyDict.Create;

   if (ord <> nil)      then kwargs['ord'] := TNumPy.ToPython(ord^);

   linalg := FhModuleNumPy.GetAttr('linalg');
   py := linalg.InvokeMethod('norm', pyargs, kwargs);

   Result := TNumPy.ToCsharp<Double>(py);

end;

function TNumPyArray.norm(x: TNDarray; ord: PChar): Double;
var
  linalg : TPythonObject;
  pyargs : TPyTuple;
  kwargs : TPyDict;
  py     : TPythonObject;
begin
   pyargs := TNumPy.ToTuple([x]);
   kwargs := TPyDict.Create;

   if (ord <> nil)      then kwargs['ord'] := TNumPy.ToPython(ord);

   linalg := FhModuleNumPy.GetAttr('linalg');
   py := linalg.InvokeMethod('norm', pyargs, kwargs);

   Result := TNumPy.ToCsharp<Double>(py);

end;

function TNumPyArray.norm(x: TNDarray; ord: Constants): Double;
var
  linalg : TPythonObject;
  pyargs : TPyTuple;
  kwargs : TPyDict;
  py     : TPythonObject;
begin
   pyargs := TNumPy.ToTuple([x]);
   kwargs := TPyDict.Create;

   if  (ord = Constants.inf)  then kwargs['ord'] := ToPython(inf)
   else                            kwargs['ord'] := ToPython(-inf);

   linalg := FhModuleNumPy.GetAttr('linalg');
   py := linalg.InvokeMethod('norm', pyargs, kwargs);

   Result := TNumPy.ToCsharp<Double>(py);

end;

(*********************************** //END //np.linalg.norm.cs *************)
(************************************************************************)

(*********************************** //START //np.linspace.cs *************)
(************************************************************************)

function TNumPyArray.linspace(start, stop: TNDarray; var step: Double; num:Integer = 50; endpoint:Boolean = true; dtype : TDtype= nil; axis: Integer = 0):TNDarray;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
   res    : TArray<TPythonObject> ;
begin
    pyargs := TNumPy.ToTuple([start,stop]);
    kwargs := TPyDict.Create;

    if (num <> 50)        then kwargs['num']      := ToPython(num);
    if (endpoint <> True) then kwargs['endpoint'] := ToPython(endpoint);
    kwargs['retstep']                             := ToPython(true); // we want the step to be returned!
    if (dtype <> nil)     then kwargs['dtype']    := ToPython(dtype);
    if (axis <> 0)        then kwargs['axis']     := ToPython(axis);

    py := FhModuleNumPy.InvokeMethod('linspace', pyargs, kwargs);

    res := py.AsArrayofPyObj;

    step := TNumPy.ToCsharp<Double>(res[0]);
    Result := TNumPy.ToCsharp<TNDarray>(res[1]);
end;

function TNumPyArray.linspace(start, stop: Double; var step: Double; num:Integer = 50; endpoint:Boolean = true; dtype : TDtype= nil; axis: Integer = 0):TNDarray;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
   res    : TArray<TPythonObject> ;
begin
    pyargs := TNumPy.ToTuple([start,stop]);
    kwargs := TPyDict.Create;

    if (num <> 50)        then kwargs['num']      := ToPython(num);
    if (endpoint <> True) then kwargs['endpoint'] := ToPython(endpoint);
    kwargs['retstep']                             := ToPython(true); // we want the step to be returned!
    if (dtype <> nil)     then kwargs['dtype']    := ToPython(dtype);
    if (axis <> 0)        then kwargs['axis']     := ToPython(axis);

    py := FhModuleNumPy.InvokeMethod('linspace', pyargs, kwargs);

    res := py.AsArrayofPyObj;

    step := TNumPy.ToCsharp<Double>(res[0]);
    Result := TNumPy.ToCsharp<TNDarray>(res[1]);
end;

(*********************************** //END //np.linspace.cs *************)
(************************************************************************)

(*********************************** //START //np.resize.cs *************)
(************************************************************************)
function TNumPyArray.resize(a: TNDarray; new_shape: Tnp_Shape): TNDarray;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
   res    : TArray<TPythonObject> ;
begin
    pyargs := TNumPy.ToTuple([a, TValue.FromShape(new_shape)]);
    kwargs := TPyDict.Create;

    py := FhModuleNumPy.InvokeMethod('resize', pyargs, kwargs);

    Result := TNumPy.ToCsharp<TNDarray>(py);

end;

(*********************************** //END //np.linspace.cs *************)
(************************************************************************)
end.
