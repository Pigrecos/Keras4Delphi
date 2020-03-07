{*******************************************************}
{                                                       }
{       Numpy Wrapper for Delphi                        }
{                                                       }
{       Copyright (C) 2020 Pigrecos(Max)                }
{                                                       }
{*******************************************************}
unit NDArray.Api;
//NDarray.gen.cs


interface
  uses System.SysUtils, System.Rtti, System.Classes, System.Generics.Collections,
       PythonEngine,
       Python.Utils,

       np.Models;

const
  null = 'null';

type
  TNDarrayGeneral = class helper for TNDArray
     public
        Function  item<T>(args: TArray<Integer>): T;
        Procedure tofile(fid : string ; sep : string ; format : string  );
        Procedure dump(ffile : string  );
        Procedure dumps;
        Function  astype(dtype : TDtype ; order  : string  = null; casting  : string  = null; subok  : PBoolean  = nil; copy  : PBoolean  = nil): TNDarray;
        Function  byteswap(inplace  : PBoolean  = nil): TNDarray;
        Function  copy(order  : string  = null): TNDarray;
        Procedure getfield(dtype : TDtype ; offset : Integer  );
        Procedure setflags(write  : PBoolean  = nil; align  : PBoolean  = nil; uic  : PBoolean  = nil);
        Procedure fill(value: TValue);
        Function  transpose(axes: TArray<Integer>): TNDarray;
        Function  flatten(order  : string  = null): TNDarray;
        Procedure __setstate__(version : Integer ; shape : Tnp_Shape ; dtype : TDtype ; isFortran : Boolean ; rawdata : string  );
        Function  reshape(newshape : Tnp_Shape ; order  : string  = null): TNDarray; overload;
        function  append(values : TNDarray ; axis  : PInteger = nil):TNDarray;
        function  asarray_chkfinite(dtype  : TDtype  = nil; order  : string  = null):TNDarray;
        function  asfarray(dtype  : TDtype  = nil):TNDarray;
        function  asfortranarray(dtype  : TDtype  = nil):TNDarray;
        function  broadcast(in1: TNDarray): TNDarray;
        function  broadcast_to(shape : Tnp_Shape ; subok : Boolean = false):TNDarray;
        function  delete(obj : Tnp_Slice ; axis  : PInteger = nil):TNDarray;
        function  expand_dims(axis: Integer): TNDarray;
        function  flip(axis: TArray<Integer>): TNDarray;
        function  fliplr: TNDarray;
        function  flipud: TNDarray;
        function  insert(obj : Integer = 0; values  : TNDarray  = nil; axis  : PInteger = nil):TNDarray;
        function  moveaxis(source, destination: TArray<Integer>): TNDarray;
        Function  ravel(order: string  = null): TNDarray;
        function  repeatt(repeats: TArray<Integer>; axis  : PInteger  = nil):TNDarray;
        function  require(dtype: TDtype; requirements: TArray<String>): TNDarray;
        function  rollaxis(axis : Integer ; start : Integer = 0):TNDarray;
        function  split(indices_or_sections: TArray<Integer>; axis: Integer = 0): TArray<TNDarray>;
        function  squeeze(axis: TArray<Integer>): TNDarray;
        function  swapaxes(axis1, axis2: Integer): TNDarray;
        function  tile(reps: TNDarray): TNDarray;
        function  trim_zeros(trim : string = 'fb'):TNDarray;
        function  unique(axis  : PInteger = nil):TNDarray; overload;
        Function  unique(return_index : Boolean ; return_inverse: Boolean; return_counts: Boolean; axis  : PInteger = nil): TArray<TNDarray>; overload;
        function  bitwise_and(x1: TNDarray; var _out, where: TNDarray): TNDarray;
        function  bitwise_or(x1: TNDarray; var _out, where: TNDarray): TNDarray;
        function  bitwise_xor(x1: TNDarray; var _out, where: TNDarray): TNDarray;
        function  common_type(array1: TNDarray): TDtype;
        procedure diag_indices_from;
        function  diagonal(offset : Integer = 0; axis1 : Integer = 0; axis2 : Integer = 1):TNDarray;
        procedure fill_diagonal(val : TValue ; wrap : Boolean = false);
        function  fv(nper : TNDarray ; pmt : TNDarray ; pv : TNDarray ; when : string = 'end'): TNDarray;
        function  i0: TNDarray;
        function  invert(var _out, where: TNDarray): TNDarray;
        function  ipmt(per :  TNDarray ; nper :  TNDarray ; pv :  TNDarray ; fv  :  TNDarray  = nil; when : string = 'end'):TNDarray;
        function  irr: Double;
        function  min_scalar_type: TDtype;
        function  mirr(finance_rate, reinvest_rate: TValue): Double;
        procedure ndenumerate;
        function  nonzero: TArray<TNDarray>;
        procedure nper(pmt :  TNDarray ; pv :  TNDarray ; fv  :  TNDarray  = nil; when : string = 'end');
        function  packbits(axis  : PInteger = nil):TNDarray;
        procedure place(mask, vals: TNDarray);
        function  pmt(nper :  TNDarray ; pv :  TNDarray ; fv  :  TNDarray  = nil; when : string = 'end'):TNDarray;
        procedure ppmt(per :  TNDarray ; nper :  TNDarray ; pv :  TNDarray ; fv  :  TNDarray  = nil; when : string = 'end');
        procedure put(ind :  TNDarray ; v :  TNDarray ; mode : string = 'raise');
        procedure put_along_axis(indices: TNDarray; values: TArray<TNDarray>; axis: Integer);
        procedure putmask(mask, values: TNDarray);
        function  pv(nper : TNDarray ; pmt : TNDarray ; fv  : TNDarray  = nil; when : string = 'end'):TNDarray;
        procedure rate(pmt :  TNDarray ; pv :  TNDarray ; fv :  TNDarray ; when : string = 'end'; guess  : PDouble  = nil; tol  : PDouble  = nil; maxiter : Integer = 100);
        function  right_shift(x2: TNDarray; var _out, where: TNDarray): TNDarray;
        function  roll(shift, axis: TArray<Integer>): TNDarray;
        function  rot90(axes : TArray<Integer>; k : Integer = 1): TNDarray;
        procedure take_along_axis(indices: TNDarray; axis: Integer);
        procedure tril_indices_from(k : Integer = 0);
        function  triu_indices_from(k : Integer = 0): TArray<TNDarray>;
        function  unpackbits(axis  : PInteger = nil):TNDarray;
        function  unravel_index(shape : Tnp_Shape ; order  : string  = null):TArray<TNDarray>;
        function  where(y, x: TNDarray): TNDarray;
        function  all(axis: TArray<Integer>; var _out  :  TNDarray  ; keepdims  : PBoolean  = nil): TNDArray<Boolean>; overload;
        Function  all: Boolean; overload;
        function  any(axis: TArray<Integer>; var _out  :  TNDarray ; keepdims  : PBoolean  = nil): TNDArray<Boolean>; overload;
        Function  any: Boolean; overload;
        function  array_repr(max_line_width : PInteger = nil; precision  : PInteger = nil; suppress_small : PBoolean = nil): string;
        procedure array_str(max_line_width : PInteger = nil; precision  : PInteger = nil; suppress_small : PBoolean = nil);
        function  dot(b: TNDarray; var _out: TNDarray): TNDarray;  overload;
        function  dot(b: TNDarray): TNDarray;overload;
        function  inner(a: TNDarray): TNDarray;
        function  kron(a: TNDarray): TNDarray;
        function  matmul(x1: TNDarray; var _out: TNDarray): TNDarray;
        function  outer(b: TNDarray; var _out: TNDarray): TNDarray;
        function  tensordot(a: TNDarray; axes: TArray<Integer>): TNDarray;
        function  trace(offset : Integer; axis2 : PInteger; axis1 : PInteger; dtype  : TDtype; var _out : TNDarray):TNDarray ;overload;
        function  trace(offset : Integer = 0; axis2 : PInteger = nil; axis1 : PInteger = nil; dtype  : TDtype  = nil):TNDarray ; overload;
        function  vdot(b: TNDarray): TNDarray;
        function  allclose(a :  TNDarray ; rtol : Double = 1E-05; atol : Double = 1E-08; equal_nan : Boolean = false): Boolean;
        function  arccos(var _out, where: TNDarray): TNDarray;
        function  arcsin(var _out, where: TNDarray): TNDarray;
        function  arctan(var _out, where: TNDarray): TNDarray;
        function  arctan2(x2: TNDarray; var _out, where: TNDarray): TNDarray;
        function  array_equal(a1: TNDarray): Boolean;
        function  array_equiv(a1: TNDarray): Boolean;
        function  cos(var _out, where: TNDarray): TNDarray;
        function  degrees(var _out, where: TNDarray): TNDarray;
        function  equal(x1: TNDarray; var _out, where: TNDarray): TNDarray;
        function  greater(x1: TNDarray; var _out, where: TNDarray): TNDarray;
        function  greater_equal(x1: TNDarray; var _out, where: TNDarray): TNDArray<Boolean>;
        function  hypot(x1: TNDarray; var _out, where: TNDarray): TNDarray;
        function  isclose(a :  TNDarray ; rtol : Double = 1E-05; atol : Double = 1E-08; equal_nan : Boolean = false):TNDarray;
        function  iscomplex: TNDarray;
        function  isfinite(var _out, where: TNDarray): TNDarray;
        function  isfortran: Boolean;
        function  isinf(var _out, where: TNDarray): TNDArray<Boolean>;
        function  isnan(var _out, where: TNDarray): TNDarray;
        function  isnat(var _out, where: TNDarray): TNDarray;
        function  isneginf(var _out: TNDarray): TNDarray;
        function  isposinf(y  :  TNDarray  = nil):TNDarray;
        function  isreal: TNDarray;
        function  less(x1: TNDarray; var _out, where: TNDarray): TNDarray;
        function  less_equal(x1: TNDarray; var _out, where: TNDarray): TNDarray;
        function  logical_and(x1: TNDarray; var _out, where: TNDarray): TNDarray;
        function  logical_not(var _out, where: TNDarray): TNDArray<Boolean>;
        function  logical_or(x1: TNDarray; var _out, where: TNDarray): TNDarray;
        function  logical_xor(x1: TNDarray; var _out, where: TNDarray): TNDArray<Boolean>;
        function  not_equal(x1: TNDarray; var _out, where: TNDarray): TNDarray;
        function  radians(var _out, where: TNDarray): TNDarray;
        function  sin(var _out, where: TNDarray): TNDarray;
        function  tan(var _out, where: TNDarray): TNDarray;

        function  arccosh(var _out, where: TNDarray): TNDarray;
        function  arcsinh(var _out, where: TNDarray): TNDarray;
        function  arctanh(var _out, where: TNDarray): TNDarray;
        function  around(decimals: Integer; var _out: TNDarray): TNDarray;
        function  ceil(var _out, where: TNDarray): TNDarray;
        function  cosh(var _out, where: TNDarray): TNDarray;
        function  cross(b :  TNDarray ; axisa : Integer = -1; axisb : Integer = -1; axisc : Integer = -1; axis  : PInteger = nil):TNDarray ;
        function  cumprod(axis: PInteger; dtype: TDtype; var _out: TNDarray): TNDarray;
        function  cumsum(axis: PInteger; dtype: TDtype; var _out: TNDarray): TNDarray;
        function  deg2rad(var _out, where: TNDarray): TNDarray;
        function  diff(n : Integer = 1; axis : Integer = -1; append  :  TNDarray  = nil; prepend  :  TNDarray  = nil):TNDarray;
        function  ediff1d(to_end : TNDarray = nil; to_begin : TNDarray = nil):TNDarray;
        function  exp(var _out, where: TNDarray): TNDarray;
        function  exp2(var _out, where: TNDarray): TNDarray;
        function  expm1(var _out, where: TNDarray): TNDarray;
        function  fix(y  :  TNDarray  = nil):TNDarray ;
        function  floor(var _out, where: TNDarray): TNDarray;
        function  gradient(_varargs: TNDarray; edge_order: PInteger; axis: TArray<Integer>): TNDarray;
        function  log(var _out, where: TNDarray): TNDarray;
        function  log10(var _out, where: TNDarray): TNDarray;
        function  log1p(var _out, where: TNDarray): TNDarray;
        function  log2(var _out, where: TNDarray): TNDarray;
        function  nancumprod(axis: PInteger; dtype: TDtype; var _out: TNDarray): TNDarray;
        function  nancumsum(axis: PInteger; dtype: TDtype; var _out: TNDarray): TNDarray;
        function  nanprod(axis: TArray<Integer>; dtype: TDtype; var _out: TNDarray; keepdims: PBoolean): TNDarray;
        function  nansum(axis: TArray<Integer>; dtype: TDtype; var _out: TNDarray; keepdims: PBoolean): TNDarray;
        function  prod(axis: TArray<Integer>; dtype: TDtype; var _out: TNDarray; keepdims: PBoolean; initial: TValue): TNDarray;
        function  rad2deg(var _out, where: TNDarray): TNDarray;
        function  rint(var _out, where: TNDarray): TNDarray;
        function  sinh(var _out, where: TNDarray): TNDarray;
        function  sum(axis: TArray<Integer>; dtype: TDtype; var _out: TNDarray; keepdims: PBoolean; initial: TValue): TNDarray;overload;
        function  sum: TNDarray; overload;
        function  tanh(var _out, where: TNDarray): TNDarray;
        function  trapz(x  :  TNDarray  = nil; dx : Double = 1.0; axis : Integer = -1): Double;
        function  trunc(var _out, where: TNDarray): TNDarray;
        function  unwrap(discont : Double = 3.141592653589793; axis : Integer = -1):TNDarray;

        function add(x1: TNDarray; var _out, where: TNDarray): TNDarray;
        function copysign(x2: TNDarray; var _out, where: TNDarray): TNDarray;
        function divide(x2: TNDarray; var _out, where: TNDarray): TNDarray;
        function float_power(x2: TNDarray; var _out, where: TNDarray): TNDarray;
        function floor_divide(x2: TNDarray; var _out, where: TNDarray): TNDarray;
        function fmod(x2: TNDarray; var _out, where: TNDarray): TNDarray;
        function frexp(out1, out2: TNDarray; var _out, where: TNDarray): TArray<TNDarray>;
        function gcd(x1: TNDarray): TNDarray;
        function lcm(x1: TNDarray): TNDarray;
        function ldexp(x2: TNDarray; var _out, where: TNDarray): TNDarray;
        function logaddexp(x1: TNDarray; var _out, where: TNDarray): TNDarray;
        function logaddexp2(x1: TNDarray; var _out, where: TNDarray): TNDarray;
        function multiply(x1: TNDarray; var _out, where: TNDarray): TNDarray;
        function negative(var _out, where: TNDarray): TNDarray;
        function nextafter(x2: TNDarray; var _out, where: TNDarray): TNDarray;
        function positive: TNDarray;
        function power(x2: TNDarray; var _out, where: TNDarray): TNDarray;
        function reciprocal(var _out, where: TNDarray): TNDarray;
        function signbit(var _out, where: TNDarray): TNDarray;
        function sinc: TNDarray;
        function spacing(var _out, where: TNDarray): TNDarray;
        function subtract(x1: TNDarray; var _out, where: TNDarray): TNDarray;
        function true_divide(x2: TNDarray; var _out, where: TNDarray): TNDarray;

        function &mod(x2: TNDarray; var _out, where: TNDarray): TNDarray;
        function absolute(var _out, where: TNDarray): TNDarray;
        function angle(deg : Boolean = false):TNDarray ;
        function argmax(axis  : PInteger; var _out  :  TNDarray):TNDarray;
        function argmin(axis: PInteger; var _out: TNDarray): TNDarray;
        function argpartition(kth: TArray<Integer>; axis: Integer = -1; kind : string = 'introselect'; order  : string  = null):TNDarray;
        function argsort(axis: integer = -1; kind : string = 'quicksort'; order  : string  = null):TNDarray;
        function argwhere: TNDarray;
        function cbrt(var _out, where: TNDarray): TNDarray;
        function clip(a_min, a_max: TNDarray; var _out: TNDarray): TNDarray;
        function conj(var _out, where: TNDarray): TNDarray;
        function convolve(v :  TNDarray ; mode : string = 'full'):TNDarray;
        function count_nonzero(axis: TArray<Integer>): TNDArray<Integer>; overload;
        Function count_nonzero: Integer; overload;
        function divmod(x2: TNDarray; var _out, where: TNDarray): TArray<TNDarray>;
        function extract(arr: TNDarray): TNDarray;
        function fabs(var _out, where: TNDarray): TNDarray;
        function flatnonzero: TNDarray;
        function fmax(x1: TNDarray; var _out, where: TNDarray): TNDarray;
        function fmin(x1: TNDarray; var _out, where: TNDarray): TNDarray;
        function heaviside(x2: TNDarray; var _out, where: TNDarray): TNDarray;
        function imag: TNDarray;
        function in1d(ar2: TNDarray; assume_unique :Boolean = false; invert : Boolean = false): TNDarray;
        function intersect1d(ar1: TNDarray; assume_unique : Boolean = false;  return_indices : Boolean = false):TArray<TNDarray>;
        function isin(test_elements: TNDarray; assume_unique : Boolean = false; invert : Boolean = false): TNDarray;
        function lexsort(axis: Integer = -1):TNDarray;
        function maximum(x1: TNDarray; var _out, where: TNDarray): TNDarray;
        function minimum(x1: TNDarray; var _out, where: TNDarray): TNDarray;
        function modf(var _out, where: TNDarray): TArray<TNDarray>;
        function msort: TNDarray;
        function nan_to_num(copy : Boolean= true):TNDarray;
        function nanargmax(axis  : PInteger = nil):TNDarray;
        function nanargmin(axis  : PInteger = nil):TNDarray;
        function nanmax(axis : TArray<Integer> ; var _out  :  TNDarray ; keepdims  : PBoolean  = nil):TNDarray;
        function nanmin(axis: TArray<Integer> ; var _out  :  TNDarray ; keepdims  : PBoolean = nil ):TNDarray ;
        function pad(pad_width: TNDarray; mode : string ; stat_length : TArray<Integer>;  constant_values : TArray<Integer>; end_values : TArray<Integer>; reflect_type : string = null):TNDarray;
        function partition(kth: TArray<Integer>; axis : Integer = -1; kind : string = 'introselect'; order  : string  = null):TNDarray;
        function percentile(q: TNDArray<Double>; axis: TArray<Integer>; var _out  :  TNDarray ; overwrite_input : Boolean = false;  interpolation : string= 'linear'; keepdims : Boolean = false):TNDArray<double>; overload;
        Function percentile(q: TNDArray<Double>; var _out  :  TNDarray  ; overwrite_input : Boolean = false;  interpolation : string = 'linear'): double; overload;
        function ptp(axis: TArray<Integer> ; var _out  :  TNDarray  ; keepdims  : PBoolean  = nil):TNDarray;
        function real: TNDarray;
        function real_if_close(tol : Double = 100):TNDarray;
        function remainder(x2: TNDarray; var _out, where: TNDarray): TNDarray;
        function searchsorted(v :  TNDarray ; side: string = 'left'; sorter  :  TNDarray  = nil): TNDArray<Integer>;
        function setdiff1d(ar2: TNDarray; assume_unique : Boolean = false): TNDarray;
        function setxor1d(ar1: TNDarray; assume_unique: Boolean = false):TNDarray;
        function sign(var _out, where: TNDarray): TNDarray;
        function sort(axis: Integer = -1; kind : string = 'quicksort'; order  : string  = null): TNDarray;
        function sort_complex: TNDarray;
        function sqrt(var _out, where: TNDarray): TNDarray;
        function square(var _out, where: TNDarray): TNDarray;
        function union1d(ar1: TNDarray): TNDarray;

        function &var(axis: TArray<Integer>; dtype  : TDtype  ; var _out  :  TNDarray  ; ddof : Integer = 0; keepdims  : PBoolean  = nil):TNDArray<double>; overload;
        Function &var(dtype  : TDtype  ; var _out  :  TNDarray ; ddof : Integer = 0): double; overload;
        function average(axis: TArray<Integer>; weights  :  TNDarray  = nil; returned : Boolean = false): TNDArray<double>;overload;
        Function average(weights  :  TNDarray  = nil; returned : Boolean = false): double; overload;
        function bincount(weights  :  TNDarray  = nil; minlength : Integer = 0):TNDarray;
        function corrcoef(y  :  TNDarray  = nil; rowvar: Boolean = true):TNDarray;
        function correlate(a :  TNDarray ; mode: string = 'valid'):TNDarray;
        function cov(y  :  TNDarray  = nil; rowvar : Boolean = true; bias : Boolean = false; ddof  : PInteger = nil; fweights  :  TNDarray  = nil; aweights  :  TNDarray  = nil):TNDarray;
        function digitize(bins :  TNDarray ; right : Boolean = false):TNDarray;

        function histogram(bins  : PInteger = nil; range: TArray<Double> = []; normed  : PBoolean  = nil; weights  :  TNDarray  = nil; density  : PBoolean  = nil):TArray<TNDarray>; overload;
        Function histogram(bins  :  TNDarray  = nil; range: TArray<Double> = []; normed  : PBoolean  = nil; weights  :  TNDarray  = nil; density  : PBoolean  = nil):TArray<TNDarray>; overload;
        Function histogram(bins : TList<string> = nil; range: TArray<Double> = []; normed  : PBoolean  = nil; weights  :  TNDarray  = nil; density  : PBoolean  = nil):TArray<TNDarray>;overload;

        function histogram_bin_edges(bins  : PInteger = nil; range: TArray<Double> = []; weights  :  TNDarray  = nil): TNDarray<Double>;overload;
        Function histogram_bin_edges(bins  :  TNDarray  = nil; range: TArray<Double> = []; weights  :  TNDarray  = nil):TNDarray<Double>;overload;
        Function histogram_bin_edges(bins: TList<string> = nil;  range: TArray<Double> = [];  weights  :  TNDarray  = nil):TNDarray<Double>;overload;

        function histogram2d(y :  TNDarray ; bins  : PInteger = nil; range: TArray<Double> = []; density  : PBoolean  = nil; normed  : PBoolean  = nil; weights  :  TNDarray  = nil):TArray<TNDarray>;overload;
        Function histogram2d(y :  TNDarray ; bins  :  TNDarray  = nil; range: TArray<Double> = []; density  : PBoolean  = nil; normed  : PBoolean  = nil; weights  :  TNDarray  = nil):TArray<TNDarray>;overload;
        Function histogram2d(y :  TNDarray ; bins : TList<string> = nil; range: TArray<Double> = []; density  : PBoolean  = nil; normed  : PBoolean  = nil; weights  :  TNDarray  = nil):TArray<TNDarray>;overload;

        function histogramdd(bins  : PInteger = nil; range: TArray<Double> = []; density  : PBoolean  = nil; normed  : PBoolean  = nil; weights  :  TNDarray  = nil):TArray<TNDarray>; overload;
        Function histogramdd(bins  :  TNDarray  = nil; range: TArray<Double> = []; density  : PBoolean  = nil; normed  : PBoolean  = nil; weights  :  TNDarray  = nil):TArray<TNDarray>;overload;
        Function histogramdd(bins : TList<string> = nil; range: TArray<Double> = []; density  : PBoolean  = nil; normed  : PBoolean  = nil; weights  :  TNDarray  = nil):TArray<TNDarray>;overload;

        function mean(axis: TArray<Integer>; dtype  : TDtype ; var _out  :  TNDarray ; keepdims  : PBoolean  = nil): TNDArray<double>; overload;
        Function mean(dtype  : TDtype  ; var _out  :  TNDarray  ): double; overload;
        function median(axis: TArray<Integer>; var _out  :  TNDarray  ; overwrite_input : Boolean = false; keepdims : Boolean = false): TNDArray<Double>;overload;
        Function median(var _out  :  TNDarray ; overwrite_input : Boolean = false): double; overload;
        function nanmean(axis: TArray<Integer>; dtype  : TDtype ; var _out  :  TNDarray  ; keepdims  : PBoolean  = nil):TNDArray<double>; overload;
        Function nanmean(dtype  : TDtype  ; var _out  :  TNDarray): double; overload;
        function nanmedian(axis: TArray<Integer>; var _out  :  TNDarray ; overwrite_input : Boolean = false; keepdims  : PBoolean  = nil): TNDArray<double>; overload;
        Function nanmedian(var _out  :  TNDarray ; overwrite_input : Boolean = false): double;overload;
        function nanpercentile(q: TNDArray<Double>; axis: TArray<Integer>; var _out  :  TNDarray ; overwrite_input : Boolean = false;  interpolation : string = 'linear'; keepdims  : PBoolean  = nil): TNDArray<double>;overload;
        Function nanpercentile(q: TNDArray<Double>; var _out  :  TNDarray  ; overwrite_input : Boolean = false; interpolation : string = 'linear') : Double;overload;
        function nanquantile(q: TNDArray<Double>; axis: TArray<Integer>; var _out  :  TNDarray ; overwrite_input: Boolean = false; interpolation : string = 'linear'; keepdims  : PBoolean  = nil): TNDArray<Double>;overload;
        Function nanquantile(q: TNDArray<Double>; var _out  :  TNDarray  ; overwrite_input : Boolean = false; interpolation : string = 'linear') : Double; overload;
        function nanstd(axis: TArray<Integer>; dtype  : TDtype ; var _out  :  TNDarray ; ddof : Integer = 0; keepdims  : PBoolean  = nil): TNDarray<double>;overload;
        Function nanstd(dtype  : TDtype  ; var _out  :  TNDarray ; ddof: Integer  = 0): double;  overload;
        function nanvar(axis: TArray<Integer>; dtype  : TDtype  ; var _out  :  TNDarray ; ddof: Integer = 0; keepdims  : PBoolean  = nil): TNDarray<double>; overload;
        Function nanvar(dtype  : TDtype  ; var _out  :  TNDarray ; ddof: Integer = 0): Double; overload;
        function quantile(q: TNDArray<Double>; axis : TArray<Integer>; var _out  :  TNDarray ; overwrite_input : Boolean = false; interpolation : string = 'linear'; keepdims: Boolean = false) : TNDArray<double>; overload;
        Function quantile(q: TNDArray<Double>; var _out  :  TNDarray  ; overwrite_input : Boolean = false;  interpolation : string = 'linear') : Double;overload;
        function std(axis:TArray<Integer>; dtype  : TDtype  ; var _out  :  TNDarray ; ddof: Integer = 0; keepdims  : PBoolean  = nil):TNDArray<double>; overload;
        Function std(dtype  : TDtype ; var _out  :  TNDarray  ; ddof : Integer = 0): Double; overload;
  end;


implementation
 uses
    np.Base,np.Utils;

Function  TNDarrayGeneral.item<T>(args: TArray<Integer>): T;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin

  pyargs := TNumPy.ToTuple([TValue.FromArray<Integer>(args)]);
  kwargs := TPyDict.Create;
  py := InvokeMethod('item', pyargs, kwargs);
  Result := TNumPy.ToCsharp<T>(py);
end;

{
Function  TNDarrayGeneral.List<T>: tolist<T>()
var
   np     : Tnp;
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  py := InvokeMethod('tolist');
  Result := np.ToCsharp<List<T>>(py);
end;
}

Procedure  TNDarrayGeneral.tofile(fid : string ; sep : string ; format : string  );
var
   pyargs : TPyTuple;
   kwargs : TPyDict;

begin
  pyargs := TNumPy.ToTuple([fid, sep, format]);
  kwargs := TPyDict.Create;
  InvokeMethod('tofile', pyargs, kwargs);
end;

Procedure  TNDarrayGeneral.dump(ffile : string  );
var
   pyargs : TPyTuple;
   kwargs : TPyDict;

begin
  pyargs := TNumPy.ToTuple([ffile]);
  kwargs := TPyDict.Create;
  InvokeMethod('dump', pyargs, kwargs);
end;

Procedure  TNDarrayGeneral.dumps;
begin
  InvokeMethod('dumps',[]);
end;

Function  TNDarrayGeneral.astype(dtype : TDtype ; order  : string; casting  : string; subok  : PBoolean; copy  : PBoolean): TNDarray;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple([dtype]);
  kwargs := TPyDict.Create;
  if (order<>null)   then   kwargs['order']  :=TNumPy.ToPython(order);
  if (casting<>null) then   kwargs['casting']:=TNumPy.ToPython(casting);
  if (subok<>nil)    then   kwargs['subok']  :=TNumPy.ToPython(subok^);
  if (copy<>nil)     then   kwargs['copy']   :=TNumPy.ToPython(copy^);
  py := InvokeMethod('astype', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function  TNDarrayGeneral.byteswap(inplace  : PBoolean  = nil): TNDarray;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple();
  kwargs := TPyDict.Create;
  if (inplace<>nil) then kwargs['inplace'] := TNumPy.ToPython(inplace^);
  py := InvokeMethod('byteswap', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function  TNDarrayGeneral.copy(order  : string  = null): TNDarray;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple();
  kwargs := TPyDict.Create;
  if (order<>null) then kwargs['order'] := TNumPy.ToPython(order);
  py := InvokeMethod('copy', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Procedure  TNDarrayGeneral.getfield(dtype : TDtype ; offset : Integer  );
var
   pyargs : TPyTuple;
   kwargs : TPyDict;

begin
  pyargs := TNumPy.ToTuple([dtype,offset]);
  kwargs := TPyDict.Create;
  InvokeMethod('getfield', pyargs, kwargs);
end;

Procedure  TNDarrayGeneral.setflags(write  : PBoolean  = nil; align  : PBoolean  = nil; uic  : PBoolean  = nil);
var
   pyargs : TPyTuple;
   kwargs : TPyDict;

begin
  pyargs := TNumPy.ToTuple();
  kwargs := TPyDict.Create;
  if (write<>nil) then kwargs['write'] := TNumPy.ToPython(write^);
  if (align<>nil) then kwargs['align'] := TNumPy.ToPython(align^);
  if (uic<>nil) then kwargs['uic'] := TNumPy.ToPython(uic^);
  InvokeMethod('setflags', pyargs, kwargs);
end;

Procedure  TNDarrayGeneral.fill(value: TValue);
var
   pyargs : TPyTuple;
   kwargs : TPyDict;

begin
  pyargs := TNumPy.ToTuple([value]);
  kwargs := TPyDict.Create;
  InvokeMethod('fill', pyargs, kwargs);
end;

Function  TNDarrayGeneral.transpose(axes: TArray<Integer>): TNDarray;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple();
  kwargs := TPyDict.Create;
  if (Length(axes) > 0) then kwargs['axes'] := TNumPy.ToPython(TValue.FromArray<Integer>(axes));
  py := InvokeMethod('transpose', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function  TNDarrayGeneral.flatten(order  : string  = null): TNDarray;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple();
  kwargs := TPyDict.Create;
  if (order<>null) then kwargs['order'] := TNumPy.ToPython(order);
  py := InvokeMethod('flatten', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;


Procedure TNDarrayGeneral.__setstate__(version : Integer ; shape : Tnp_Shape ; dtype : TDtype ; isFortran : Boolean ; rawdata : string  );
var
   pyargs : TPyTuple;
   kwargs : TPyDict;

begin
  pyargs := TNumPy.ToTuple([version, TValue.FromShape(shape),dtype,isFortran,rawdata]);
  kwargs := TPyDict.Create;
  InvokeMethod('__setstate__', pyargs, kwargs);
end;

Function  TNDarrayGeneral.reshape(newshape : Tnp_Shape ; order  : string  = null): TNDarray;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple([TValue.FromShape(newshape)]);
  kwargs := TPyDict.Create;
  if (order<>null) then kwargs['order'] := TNumPy.ToPython(order);
  py := InvokeMethod('reshape', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function  TNDarrayGeneral.ravel(order: string  = null): TNDarray;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple();
  kwargs := TPyDict.Create;
  if (order<>null) then kwargs['order'] := TNumPy.ToPython(order);
  py := InvokeMethod('ravel', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function  TNDarrayGeneral.moveaxis(source: TArray<Integer>; destination: TArray<Integer>):TNDarray ;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple([tvalue.FromArray<Integer>(source),
                        tvalue.Fromarray<Integer>(destination)]);
  kwargs := TPyDict.Create;
  py := InvokeMethod('moveaxis', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;


Function  TNDarrayGeneral.rollaxis(axis : Integer ; start : Integer = 0):TNDarray ;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple([axis]);
  kwargs := TPyDict.Create;
  if (start<>0) then kwargs['start'] := TNumPy.ToPython(start);
  py := InvokeMethod('rollaxis', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function  TNDarrayGeneral.swapaxes(axis1, axis2: Integer):TNDarray ;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple([axis1,axis2]);
  kwargs := TPyDict.Create;
  py := InvokeMethod('swapaxes', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function  TNDarrayGeneral.broadcast(in1: TNDarray):TNDarray ;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple([in1]);
  kwargs := TPyDict.Create;
  py := InvokeMethod('broadcast', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function  TNDarrayGeneral.broadcast_to(shape : Tnp_Shape ; subok : Boolean = false):TNDarray ;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple([TValue.FromShape(shape)]);
  kwargs := TPyDict.Create;
  if (subok<>false) then kwargs['subok'] := TNumPy.ToPython(subok);
  py := InvokeMethod('broadcast_to', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNDarrayGeneral.expand_dims(axis : Integer ):TNDarray ;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple([axis]);
  kwargs := TPyDict.Create;
  py := InvokeMethod('expand_dims', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function  TNDarrayGeneral.squeeze(axis: TArray<Integer>):TNDarray ;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple();
  kwargs := TPyDict.Create;
  if (Length(axis) > 0) then kwargs['axis'] := TNumPy.ToPython(TValue.FromArray<Integer>(axis));
  py := InvokeMethod('squeeze', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNDarrayGeneral.asfarray(dtype  : TDtype  = nil):TNDarray ;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple();
  kwargs := TPyDict.Create;
  if (dtype<>nil) then kwargs['dtype'] := TNumPy.ToPython(dtype);
  py := InvokeMethod('asfarray', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNDarrayGeneral.asfortranarray(dtype  : TDtype  = nil):TNDarray ;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple();
  kwargs := TPyDict.Create;
  if (dtype<>nil) then kwargs['dtype'] := TNumPy.ToPython(dtype);
  py := InvokeMethod('asfortranarray', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNDarrayGeneral.asarray_chkfinite(dtype  : TDtype  = nil; order  : string  = null):TNDarray ;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple();
  kwargs := TPyDict.Create;
  if (dtype<>nil) then kwargs['dtype'] := TNumPy.ToPython(dtype);
  if (order<>null) then kwargs['order'] := TNumPy.ToPython(order);
  py := InvokeMethod('asarray_chkfinite', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNDarrayGeneral.require(dtype : TDtype ; requirements : TArray<String>):TNDarray ;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple([dtype]);
  kwargs := TPyDict.Create;
  if (Length(requirements) > 0) then kwargs['requirements'] := TNumPy.ToPython(TValue.FromArray<string>(requirements));
  py := InvokeMethod('require', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function  TNDarrayGeneral.split(indices_or_sections: TArray<Integer>; axis: Integer = 0): TArray<TNDarray> ;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple([tvalue.FromArray<Integer>(indices_or_sections) ]);
  kwargs := TPyDict.Create;
  if (axis<>0) then kwargs['axis'] := TNumPy.ToPython(axis);
  py := InvokeMethod('split', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TArray<TNDarray>>(py);
end;

Function TNDarrayGeneral.tile(reps : TNDarray  ):TNDarray ;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple([reps]);
  kwargs := TPyDict.Create;
  py := InvokeMethod('tile', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNDarrayGeneral.repeatt(repeats: TArray<Integer>; axis  : PInteger  = nil):TNDarray ;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple([tvalue.FromArray<Integer>(repeats)]);
  kwargs := TPyDict.Create;
  if (axis<>nil) then kwargs['axis'] := TNumPy.ToPython(axis^);
  py := InvokeMethod('repeat', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNDarrayGeneral.delete(obj : Tnp_Slice ; axis  : PInteger = nil):TNDarray ;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple([TValue.FromSlice(obj) ]);
  kwargs := TPyDict.Create;
  if (axis<>nil) then kwargs['axis'] := TNumPy.ToPython(axis^);
  py := InvokeMethod('delete', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNDarrayGeneral.insert(obj : Integer = 0; values  : TNDarray  = nil; axis  : PInteger = nil):TNDarray ;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple();
  kwargs := TPyDict.Create;
  if (obj<>0) then kwargs['obj'] := TNumPy.ToPython(obj);
  if (values<>nil) then kwargs['values'] := TNumPy.ToPython(values);
  if (axis<>nil) then kwargs['axis'] := TNumPy.ToPython(axis^);
  py := InvokeMethod('insert', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNDarrayGeneral.append(values : TNDarray ; axis  : PInteger = nil):TNDarray ;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple([values]);
  kwargs := TPyDict.Create;
  if (axis<>nil) then kwargs['axis'] := TNumPy.ToPython(axis^);
  py := InvokeMethod('append', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNDarrayGeneral.trim_zeros(trim : string = 'fb'):TNDarray ;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple();
  kwargs := TPyDict.Create;
  if (trim<>'fb') then kwargs['trim'] := TNumPy.ToPython(trim);
  py := InvokeMethod('trim_zeros', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNDarrayGeneral.unique(axis  : PInteger = nil):TNDarray ;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple();
  kwargs := TPyDict.Create;
  if (axis<>nil) then kwargs['axis'] := TNumPy.ToPython(axis^);
  py := InvokeMethod('unique', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function  TNDarrayGeneral.unique(return_index : Boolean ; return_inverse: Boolean; return_counts: Boolean; axis  : PInteger = nil): TArray<TNDarray>;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple();
  kwargs := TPyDict.Create;
  if (return_index) then kwargs['return_index'] := TNumPy.ToPython(return_index);
  if (return_inverse) then kwargs['return_inverse'] := TNumPy.ToPython(return_inverse);
  if (return_counts) then kwargs['return_counts'] := TNumPy.ToPython(return_counts);
  if (axis<>nil) then kwargs['axis'] := TNumPy.ToPython(axis^);
  py := InvokeMethod('unique', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TArray<TNDarray>>(py);
end;

Function TNDarrayGeneral.flip(axis: TArray<Integer>):TNDarray ;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple();
  kwargs := TPyDict.Create;
  if (Length(axis)> 0) then kwargs['axis'] := TNumPy.ToPython(TValue.FromArray<Integer>(axis));
  py := InvokeMethod('flip', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNDarrayGeneral.fliplr():TNDarray ;
var
   py     : TPythonObject;
begin
  py := InvokeMethod('fliplr',[]);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNDarrayGeneral.flipud():TNDarray ;
var
   py     : TPythonObject;
begin
  py := InvokeMethod('flipud',[]);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNDarrayGeneral.roll(shift: TArray<Integer>; axis :  TArray<Integer>):TNDarray ;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple([tvalue.FromArray<Integer>(shift)]);
  kwargs := TPyDict.Create;
  if (Length(axis) > 0) then kwargs['axis'] := TNumPy.ToPython(TValue.FromArray<Integer>(axis));
  py := InvokeMethod('roll', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNDarrayGeneral.rot90(axes : TArray<Integer>; k : Integer = 1):TNDarray ;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple();
  kwargs := TPyDict.Create;
  if (k<>1) then kwargs['k'] := TNumPy.ToPython(k);
  if (Length(axes) > 0) then kwargs['axes'] := TNumPy.ToPython(TValue.FromArray<Integer>(axes));
  py := InvokeMethod('rot90', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNDarrayGeneral.bitwise_and(x1: TNDarray; var _out  : TNDarray  ; var where  : TNDarray ):TNDarray ;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple([x1]);
  kwargs := TPyDict.Create;
  if (_out <> nil) then kwargs['out'] := TNumPy.ToPython(_out);
  if (where <> nil) then kwargs['where'] := TNumPy.ToPython(where);
  py := InvokeMethod('bitwise_and', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNDarrayGeneral.bitwise_or(x1: TNDarray; var _out  : TNDarray  ; var where  : TNDarray ):TNDarray ;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple([x1]);
  kwargs := TPyDict.Create;
  if (_out <> nil) then kwargs['out'] := TNumPy.ToPython(_out);
  if (where <> nil) then kwargs['where'] := TNumPy.ToPython(where);
  py := InvokeMethod('bitwise_or', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNDarrayGeneral.bitwise_xor(x1: TNDarray; var _out  : TNDarray ; var where  : TNDarray):TNDarray ;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple([x1]);
  kwargs := TPyDict.Create;
  if (_out <> nil) then kwargs['out'] := TNumPy.ToPython(_out);
  if (where <> nil) then kwargs['where'] := TNumPy.ToPython(where);
  py := InvokeMethod('bitwise_xor', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNDarrayGeneral.invert(var _out  : TNDarray ; var where  : TNDarray ):TNDarray ;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple();
  kwargs := TPyDict.Create;
  if (_out <> nil) then kwargs['out'] := TNumPy.ToPython(_out);
  if (where <> nil) then kwargs['where'] := TNumPy.ToPython(where);
  py := InvokeMethod('invert', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNDarrayGeneral.right_shift(x2: TNDarray; var _out  : TNDarray ; var where  : TNDarray ):TNDarray ;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple([x2]);
  kwargs := TPyDict.Create;
  if (_out <> nil) then kwargs['out'] := TNumPy.ToPython(_out);
  if (where <> nil) then kwargs['where'] := TNumPy.ToPython(where);
  py := InvokeMethod('right_shift', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNDarrayGeneral.packbits(axis  : PInteger = nil):TNDarray ;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple();
  kwargs := TPyDict.Create;
  if (axis<>nil) then kwargs['axis'] := TNumPy.ToPython(axis^);
  py := InvokeMethod('packbits', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNDarrayGeneral.unpackbits(axis  : PInteger = nil):TNDarray ;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple();
  kwargs := TPyDict.Create;
  if (axis<>nil) then kwargs['axis'] := TNumPy.ToPython(axis^);
  py := InvokeMethod('unpackbits', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function  TNDarrayGeneral.min_scalar_type: TDtype;
var
   py     : TPythonObject;
begin
  py := InvokeMethod('min_scalar_type',[]);
  Result := TNumPy.ToCsharp<TDtype>(py);
end;

Function  TNDarrayGeneral.common_type(array1: TNDarray): TDtype;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple([array1]);
  kwargs := TPyDict.Create;
  py := InvokeMethod('common_type', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TDtype>(py);
end;

Function TNDarrayGeneral.i0:TNDarray ;
var
  py     : TPythonObject;
begin
    py := InvokeMethod('i0',[]);
    Result := TNumPy.ToCsharp<TNDarray>(py);
end;

function TNDarrayGeneral.fv(nper : TNDarray ; pmt : TNDarray ; pv : TNDarray ; when : string = 'end'): TNDarray ;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple([nper,pmt,pv]);
  kwargs := TPyDict.Create;
  if (when<>'end') then kwargs['when'] := TNumPy.ToPython(when);
  py := InvokeMethod('fv', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNDarrayGeneral.pv(nper : TNDarray ; pmt : TNDarray ; fv  : TNDarray  = nil; when : string = 'end'):TNDarray ;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple([nper,pmt]);
  kwargs := TPyDict.Create;
  if (fv<>nil) then kwargs['fv'] := TNumPy.ToPython(fv);
  if (when<>'end') then kwargs['when'] := TNumPy.ToPython(when);
  py := InvokeMethod('pv', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNDarrayGeneral.pmt(nper :  TNDarray ; pv :  TNDarray ; fv  :  TNDarray  = nil; when : string = 'end'):TNDarray ;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple([nper,pv]);
  kwargs := TPyDict.Create;
  if (fv<>nil) then kwargs['fv'] := TNumPy.ToPython(fv);
  if (when<>'end') then kwargs['when'] := TNumPy.ToPython(when);
  py := InvokeMethod('pmt', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Procedure  TNDarrayGeneral.ppmt(per :  TNDarray ; nper :  TNDarray ; pv :  TNDarray ; fv  :  TNDarray  = nil; when : string = 'end');
var
   pyargs : TPyTuple;
   kwargs : TPyDict;

begin
  pyargs := TNumPy.ToTuple([per,nper,pv]);
  kwargs := TPyDict.Create;
  if (fv<>nil) then kwargs['fv'] := TNumPy.ToPython(fv);
  if (when<>'end') then kwargs['when'] := TNumPy.ToPython(when);
  InvokeMethod('ppmt', pyargs, kwargs);
end;

Function TNDarrayGeneral.ipmt(per :  TNDarray ; nper :  TNDarray ; pv :  TNDarray ; fv  :  TNDarray  = nil; when : string = 'end'):TNDarray ;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple([per,nper,pv]);
  kwargs := TPyDict.Create;
  if (fv<>nil) then kwargs['fv'] := TNumPy.ToPython(fv);
  if (when<>'end') then kwargs['when'] := TNumPy.ToPython(when);
  py := InvokeMethod('ipmt', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function  TNDarrayGeneral.irr: Double;
var
   py     : TPythonObject;
begin
  py := InvokeMethod('irr',[]);
  Result := TNumPy.ToCsharp<Double>(py);
end;

Function  TNDarrayGeneral.mirr(finance_rate: TValue; reinvest_rate: TValue): Double;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple([finance_rate,reinvest_rate]);
  kwargs := TPyDict.Create;
  py := InvokeMethod('mirr', pyargs, kwargs);
  Result := TNumPy.ToCsharp<double>(py);
end;

Procedure  TNDarrayGeneral.nper(pmt :  TNDarray ; pv :  TNDarray ; fv  :  TNDarray  = nil; when : string = 'end') ;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;

begin
  pyargs := TNumPy.ToTuple([pmt,pv]);
  kwargs := TPyDict.Create;
  if (fv<>nil) then kwargs['fv'] := TNumPy.ToPython(fv);
  if (when<>'end') then kwargs['when'] := TNumPy.ToPython(when);
  InvokeMethod('nper', pyargs, kwargs);
end;

Procedure  TNDarrayGeneral.rate(pmt :  TNDarray ; pv :  TNDarray ; fv :  TNDarray ; when : string = 'end'; guess  : PDouble  = nil; tol  : PDouble  = nil; maxiter : Integer = 100);
var
   pyargs : TPyTuple;
   kwargs : TPyDict;

begin
  pyargs := TNumPy.ToTuple([pmt,pv,fv]);
  kwargs := TPyDict.Create;
  if (when<>'end') then kwargs['when'] := TNumPy.ToPython(when);
  if (guess<>nil) then kwargs['guess'] := TNumPy.ToPython(guess^);
  if (tol<>nil) then kwargs['tol'] := TNumPy.ToPython(tol^);
  if (maxiter<>100) then kwargs['maxiter'] := TNumPy.ToPython(maxiter);
  InvokeMethod('rate', pyargs, kwargs);
end;

Function  TNDarrayGeneral.nonzero: TArray<TNDarray>;
var
   py     : TPythonObject;
begin
  py := InvokeMethod('nonzero',[]);
  Result := TNumPy.ToCsharp<TArray<TNDarray> >(py);
end;

Function TNDarrayGeneral.where(y :  TNDarray ; x :  TNDarray  ):TNDarray ;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple([y,x]);
  kwargs := TPyDict.Create;
  py := InvokeMethod('where', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function  TNDarrayGeneral.unravel_index(shape : Tnp_Shape ; order  : string  = null):TArray<TNDarray>;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple( [TValue.FromShape( shape)] );
  kwargs := TPyDict.Create;
  if (order<>null) then kwargs['order'] := TNumPy.ToPython(order);
  py := InvokeMethod('unravel_index', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TArray<TNDarray>>(py);
end;

Procedure  TNDarrayGeneral.diag_indices_from;
begin
  InvokeMethod('diag_indices_from',[]);
end;

Procedure  TNDarrayGeneral.tril_indices_from(k : Integer = 0);
var
   pyargs : TPyTuple;
   kwargs : TPyDict;

begin
  pyargs := TNumPy.ToTuple();
  kwargs := TPyDict.Create;
  if (k<>0) then kwargs['k'] := TNumPy.ToPython(k);
  InvokeMethod('tril_indices_from', pyargs, kwargs);
end;

Function  TNDarrayGeneral.triu_indices_from(k : Integer = 0): TArray<TNDarray>;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple();
  kwargs := TPyDict.Create;
  if (k<>0) then kwargs['k'] := TNumPy.ToPython(k);
  py := InvokeMethod('triu_indices_from', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TArray<TNDarray>>(py);
end;

Procedure  TNDarrayGeneral.take_along_axis(indices :  TNDarray ; axis : Integer  );
var
   pyargs : TPyTuple;
   kwargs : TPyDict;

begin
  pyargs := TNumPy.ToTuple([indices,axis]);
  kwargs := TPyDict.Create;
  InvokeMethod('take_along_axis', pyargs, kwargs);
end;

Function TNDarrayGeneral.diagonal(offset : Integer = 0; axis1 : Integer = 0; axis2 : Integer = 1):TNDarray ;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple();
  kwargs := TPyDict.Create;
  if (offset<>0) then kwargs['offset'] := TNumPy.ToPython(offset);
  if (axis1<>0) then kwargs['axis1'] := TNumPy.ToPython(axis1);
  if (axis2<>1) then kwargs['axis2'] := TNumPy.ToPython(axis2);
  py := InvokeMethod('diagonal', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Procedure  TNDarrayGeneral.place(mask :  TNDarray ; vals :  TNDarray  );
var
   pyargs : TPyTuple;
   kwargs : TPyDict;

begin
  pyargs := TNumPy.ToTuple([mask,vals]);
  kwargs := TPyDict.Create;
  InvokeMethod('place', pyargs, kwargs);
end;

Procedure  TNDarrayGeneral.put(ind :  TNDarray ; v :  TNDarray ; mode : string = 'raise') ;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;

begin
  pyargs := TNumPy.ToTuple([ind,v]);
  kwargs := TPyDict.Create;
  if (mode<>'raise') then kwargs['mode'] := TNumPy.ToPython(mode);
  InvokeMethod('put', pyargs, kwargs);
end;

Procedure  TNDarrayGeneral.put_along_axis(indices :  TNDarray ;  values: TArray<TNDarray>; axis : Integer  );
var
   pyargs : TPyTuple;
   kwargs : TPyDict;

begin
  pyargs := TNumPy.ToTuple([indices,tvalue.FromArray<TNDarray>(values),axis]);
  kwargs := TPyDict.Create;
  InvokeMethod('put_along_axis', pyargs, kwargs);
end;

Procedure  TNDarrayGeneral.putmask(mask :  TNDarray ; values :  TNDarray  );
var
   pyargs : TPyTuple;
   kwargs : TPyDict;

begin
  pyargs := TNumPy.ToTuple([mask,values]);
  kwargs := TPyDict.Create;
  InvokeMethod('putmask', pyargs, kwargs);
end;

Procedure  TNDarrayGeneral.fill_diagonal(val : TValue ; wrap : Boolean = false) ;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;

begin
  pyargs := TNumPy.ToTuple([val]);
  kwargs := TPyDict.Create;
  if (wrap<>false) then kwargs['wrap'] := TNumPy.ToPython(wrap);
  InvokeMethod('fill_diagonal', pyargs, kwargs);
end;

(*
Procedure  TNDarrayGeneral.nditer(string[] flags = null, list of list of str op_flags = null, dtype or tuple of dtype(s) op_dtypes = null, order  : string  = null; casting  : string  = null; list of list of ints op_axes = null, tuple of itershape  : ints  = nil; buffersize  : PInteger = nil);
var
   np     : Tnp;
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple();
  kwargs := TPyDict.Create;
  if (flags<>null) then kwargs['flags'] := TNumPy.ToPython(flags);
  if (op_flags<>null) then kwargs['op_flags'] := TNumPy.ToPython(op_flags);
  if (op_dtypes<>null) then kwargs['op_dtypes'] := TNumPy.ToPython(op_dtypes);
  if (order<>null) then kwargs['order'] := TNumPy.ToPython(order);
  if (casting<>null) then kwargs['casting'] := TNumPy.ToPython(casting);
  if (op_axes<>null) then kwargs['op_axes'] := TNumPy.ToPython(op_axes);
  if (itershape<>null) then kwargs['itershape'] := TNumPy.ToPython(itershape);
  if (buffersize<>null) then kwargs['buffersize'] := TNumPy.ToPython(buffersize);
  py := InvokeMethod('nditer', pyargs, kwargs);
end;
*)
Procedure  TNDarrayGeneral.ndenumerate;
begin
  InvokeMethod('ndenumerate',[]);
end;
(*
Function  TNDarrayGeneral.tuple of nditer nested_iters(axes: TArray<Integer>)
var
   np     : Tnp;
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple();
  kwargs := TPyDict.Create;
  if (axes<>null) then kwargs['axes'] := TNumPy.ToPython(axes);
  py := InvokeMethod('nested_iters', pyargs, kwargs);
  Result := TNumPy.ToCsharp<tuple of nditer>(py);
end;
*)
(*
Function  TNDarrayGeneral.string array2string(int max_line_width = null, precision  : PInteger = nil; bool suppress_small = null, string separator = ' ', string prefix = '', string suffix = '', dict of formatter  : callables  = nil; threshold  : PInteger = nil; edgeitems  : PInteger = nil; sign  : string  = null; floatmode  : string  = null; string or legacy  : False  = nil);
var
   np     : Tnp;
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple();
  kwargs := TPyDict.Create;
  if (max_line_width<>null) then kwargs['max_line_width'] := TNumPy.ToPython(max_line_width);
  if (precision<>null) then kwargs['precision'] := TNumPy.ToPython(precision);
  if (suppress_small<>null) then kwargs['suppress_small'] := TNumPy.ToPython(suppress_small);
  if (separator<>' ') then kwargs['separator'] := TNumPy.ToPython(separator);
  if (prefix<>'') then kwargs['prefix'] := TNumPy.ToPython(prefix);
  if (suffix<>'') then kwargs['suffix'] := TNumPy.ToPython(suffix);
  if (formatter<>null) then kwargs['formatter'] := TNumPy.ToPython(formatter);
  if (threshold<>null) then kwargs['threshold'] := TNumPy.ToPython(threshold);
  if (edgeitems<>null) then kwargs['edgeitems'] := TNumPy.ToPython(edgeitems);
  if (sign<>null) then kwargs['sign'] := TNumPy.ToPython(sign);
  if (floatmode<>null) then kwargs['floatmode'] := TNumPy.ToPython(floatmode);
  if (legacy<>null) then kwargs['legacy'] := TNumPy.ToPython(legacy);
  py := InvokeMethod('array2string', pyargs, kwargs);
  Result := TNumPy.ToCsharp<string>(py);
end;
*)

Function  TNDarrayGeneral.array_repr(max_line_width : PInteger = nil; precision  : PInteger = nil; suppress_small : PBoolean = nil): string;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple();
  kwargs := TPyDict.Create;
  if (max_line_width<>nil) then kwargs['max_line_width'] := TNumPy.ToPython(max_line_width^);
  if (precision<>nil) then kwargs['precision'] := TNumPy.ToPython(precision^);
  if (suppress_small<>nil) then kwargs['suppress_small'] := TNumPy.ToPython(suppress_small^);
  py := InvokeMethod('array_repr', pyargs, kwargs);
  Result := TNumPy.ToCsharp<string>(py);
end;

Procedure  TNDarrayGeneral.array_str(max_line_width : PInteger = nil; precision  : PInteger = nil; suppress_small : PBoolean = nil) ;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;

begin
  pyargs := TNumPy.ToTuple();
  kwargs := TPyDict.Create;
  if (max_line_width<>nil) then kwargs['max_line_width'] := TNumPy.ToPython(max_line_width^);
  if (precision<>nil) then kwargs['precision'] := TNumPy.ToPython(precision^);
  if (suppress_small<>nil) then kwargs['suppress_small'] := TNumPy.ToPython(suppress_small^);
  InvokeMethod('array_str', pyargs, kwargs);
end;

Function TNDarrayGeneral.dot(b :  TNDarray ; var _out  :  TNDarray{  = nil}):TNDarray ;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple([b]);
  kwargs := TPyDict.Create;
  if (_out <> nil) then kwargs['out'] := TNumPy.ToPython(_out);
  py := InvokeMethod('dot', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNDarrayGeneral.dot(b :  TNDarray ):TNDarray ;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple([b]);
  kwargs := TPyDict.Create;

  py := InvokeMethod('dot', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNDarrayGeneral.vdot(b :  TNDarray  ):TNDarray ;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple([b]);
  kwargs := TPyDict.Create;
  py := InvokeMethod('vdot', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNDarrayGeneral.inner(a :  TNDarray  ):TNDarray ;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple([a]);
  kwargs := TPyDict.Create;
  py := InvokeMethod('inner', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNDarrayGeneral.outer(b :  TNDarray ; var _out  :  TNDarray{  = nil}):TNDarray ;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple([b]);
  kwargs := TPyDict.Create;
  if (_out <> nil) then kwargs['out'] := TNumPy.ToPython(_out);
  py := InvokeMethod('outer', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNDarrayGeneral.matmul(x1: TNDarray; var _out  :  TNDarray{  = nil}):TNDarray ;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple([x1]);
  kwargs := TPyDict.Create;
  if (_out <> nil) then kwargs['out'] := TNumPy.ToPython(_out);
  py := InvokeMethod('matmul', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNDarrayGeneral.tensordot(a :  TNDarray ; axes : TArray<Integer>):TNDarray ;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple([a]);
  kwargs := TPyDict.Create;
  if (Length(axes) >0) then kwargs['axes'] := TNumPy.ToPython(TValue.FromArray<Integer>(axes));
  py := InvokeMethod('tensordot', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNDarrayGeneral.kron(a :  TNDarray  ):TNDarray ;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple([a]);
  kwargs := TPyDict.Create;
  py := InvokeMethod('kron', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNDarrayGeneral.trace(offset : Integer; axis2 : PInteger; axis1 : PInteger; dtype  : TDtype; var _out : TNDarray):TNDarray ;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple();
  kwargs := TPyDict.Create;
  if (offset<>0) then kwargs['offset'] := TNumPy.ToPython(offset);
  if (axis2<>nil) then kwargs['axis2'] := TNumPy.ToPython(axis2^);
  if (axis1<>nil) then kwargs['axis1'] := TNumPy.ToPython(axis1^);
  if (dtype<>nil) then kwargs['dtype'] := TNumPy.ToPython(dtype);
  if (_out <> nil) then kwargs['out'] := TNumPy.ToPython(_out);
  py := InvokeMethod('trace', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNDarrayGeneral.trace(offset : Integer = 0; axis2 : PInteger = nil; axis1 : PInteger = nil; dtype  : TDtype  = nil):TNDarray ;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple();
  kwargs := TPyDict.Create;
  if (offset<>0) then kwargs['offset'] := TNumPy.ToPython(offset);
  if (axis2<>nil) then kwargs['axis2'] := TNumPy.ToPython(axis2^);
  if (axis1<>nil) then kwargs['axis1'] := TNumPy.ToPython(axis1^);
  if (dtype<>nil) then kwargs['dtype'] := TNumPy.ToPython(dtype);

  py := InvokeMethod('trace', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;


Function  TNDarrayGeneral.all(axis: TArray<Integer>; var _out  :  TNDarray  ; keepdims  : PBoolean  = nil): TNDArray<Boolean>;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple();
  kwargs := TPyDict.Create;
  if (Length(axis) > 0) then kwargs['axis'] := TNumPy.ToPython(TValue.FromArray<Integer>(axis));
  if (_out <> nil) then kwargs['out'] := TNumPy.ToPython(_out);
  if (keepdims<>nil) then kwargs['keepdims'] := TNumPy.ToPython(keepdims^);
  py := InvokeMethod('all', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDArray<Boolean>>(py);
end;

Function  TNDarrayGeneral.all: Boolean;
var
   py     : TPythonObject;
begin
  py := InvokeMethod('all',[]);
  Result := TNumPy.ToCsharp<Boolean>(py);
end;

Function  TNDarrayGeneral.any(axis: TArray<Integer>; var _out  :  TNDarray ; keepdims  : PBoolean  = nil): TNDArray<Boolean>;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple();
  kwargs := TPyDict.Create;
  if (length(axis) > 0) then kwargs['axis'] := TNumPy.ToPython(TValue.FromArray<integer>(axis));
  if (_out <> nil) then kwargs['out'] := TNumPy.ToPython(_out);
  if (keepdims<>nil) then kwargs['keepdims'] := TNumPy.ToPython(keepdims^);
  py := InvokeMethod('any', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDArray<Boolean>>(py);
end;

Function  TNDarrayGeneral.any: Boolean;
var
   py     : TPythonObject;
begin
  py := InvokeMethod('any',[]);
  Result := TNumPy.ToCsharp<Boolean>(py);
end;

Function TNDarrayGeneral.isfinite(var _out  :  TNDarray  ; var where  :  TNDarray  ):TNDarray ;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple();
  kwargs := TPyDict.Create;
  if (_out <> nil) then kwargs['out'] := TNumPy.ToPython(_out);
  if (where <> nil) then kwargs['where'] := TNumPy.ToPython(where);
  py := InvokeMethod('isfinite', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function  TNDarrayGeneral.isinf(var _out  :  TNDarray  ; var where  :  TNDarray  ): TNDArray<Boolean> ;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple();
  kwargs := TPyDict.Create;
  if (_out <> nil) then kwargs['out'] := TNumPy.ToPython(_out);
  if (where <> nil) then kwargs['where'] := TNumPy.ToPython(where);
  py := InvokeMethod('isinf', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDArray<Boolean>>(py);
end;

Function TNDarrayGeneral.isnan(var _out  :  TNDarray  ; var where  :  TNDarray  ):TNDarray ;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple();
  kwargs := TPyDict.Create;
  if (_out <> nil) then kwargs['out'] := TNumPy.ToPython(_out);
  if (where <> nil) then kwargs['where'] := TNumPy.ToPython(where);
  py := InvokeMethod('isnan', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNDarrayGeneral.isnat(var _out  :  TNDarray ; var where  :  TNDarray ):TNDarray ;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple();
  kwargs := TPyDict.Create;
  if (_out <> nil) then kwargs['out'] := TNumPy.ToPython(_out);
  if (where <> nil) then kwargs['where'] := TNumPy.ToPython(where);
  py := InvokeMethod('isnat', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNDarrayGeneral.isneginf(var _out  :  TNDarray ):TNDarray ;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple();
  kwargs := TPyDict.Create;
  if (_out <> nil) then kwargs['out'] := TNumPy.ToPython(_out);
  py := InvokeMethod('isneginf', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNDarrayGeneral.isposinf(y  :  TNDarray  = nil):TNDarray ;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple();
  kwargs := TPyDict.Create;
  if (y<>nil) then kwargs['y'] := TNumPy.ToPython(y);
  py := InvokeMethod('isposinf', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNDarrayGeneral.iscomplex():TNDarray ;
var
   py     : TPythonObject;
begin
  py := InvokeMethod('iscomplex',[]);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function  TNDarrayGeneral.isfortran: Boolean;
var
   py     : TPythonObject;
begin
  py := InvokeMethod('isfortran',[]);
  Result := TNumPy.ToCsharp<Boolean>(py);
end;

Function TNDarrayGeneral.isreal():TNDarray ;
var
   py     : TPythonObject;
begin
  py := InvokeMethod('isreal',[]);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNDarrayGeneral.logical_and(x1: TNDarray; var _out  :  TNDarray  ; var where  :  TNDarray ):TNDarray ;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple([x1]);
  kwargs := TPyDict.Create;
  if (_out <> nil) then kwargs['out'] := TNumPy.ToPython(_out);
  if (where <> nil) then kwargs['where'] := TNumPy.ToPython(where);
  py := InvokeMethod('logical_and', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNDarrayGeneral.logical_or(x1: TNDarray; var _out  :  TNDarray; var where  :  TNDarray):TNDarray ;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple([x1]);
  kwargs := TPyDict.Create;
  if (_out <> nil) then kwargs['out'] := TNumPy.ToPython(_out);
  if (where <> nil) then kwargs['where'] := TNumPy.ToPython(where);
  py := InvokeMethod('logical_or', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function  TNDarrayGeneral.logical_not(var _out  :  TNDarray ; var where  :  TNDarray): TNDArray<Boolean> ;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple();
  kwargs := TPyDict.Create;
  if (_out <> nil) then kwargs['out'] := TNumPy.ToPython(_out);
  if (where <> nil) then kwargs['where'] := TNumPy.ToPython(where);
  py := InvokeMethod('logical_not', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDArray<Boolean>>(py);
end;

Function  TNDarrayGeneral.logical_xor(x1: TNDarray; var _out  :  TNDarray ; var where  :  TNDarray): TNDArray<Boolean> ;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple([x1]);
  kwargs := TPyDict.Create;
  if (_out <> nil) then kwargs['out'] := TNumPy.ToPython(_out);
  if (where <> nil) then kwargs['where'] := TNumPy.ToPython(where);
  py := InvokeMethod('logical_xor', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDArray<Boolean>>(py);
end;

Function  TNDarrayGeneral.allclose(a :  TNDarray ; rtol : Double = 1E-05; atol : Double = 1E-08; equal_nan : Boolean = false): Boolean;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple([a]);
  kwargs := TPyDict.Create;
  if (rtol<>1E-05) then kwargs['rtol'] := TNumPy.ToPython(rtol);
  if (atol<>1E-08) then kwargs['atol'] := TNumPy.ToPython(atol);
  if (equal_nan<>false) then kwargs['equal_nan'] := TNumPy.ToPython(equal_nan);
  py := InvokeMethod('allclose', pyargs, kwargs);
  Result := TNumPy.ToCsharp<Boolean>(py);
end;

Function TNDarrayGeneral.isclose(a :  TNDarray ; rtol : Double = 1E-05; atol : Double = 1E-08; equal_nan : Boolean = false):TNDarray;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple([a]);
  kwargs := TPyDict.Create;
  if (rtol<>1e-05) then kwargs['rtol'] := TNumPy.ToPython(rtol);
  if (atol<>1e-08) then kwargs['atol'] := TNumPy.ToPython(atol);
  if (equal_nan<>false) then kwargs['equal_nan'] := TNumPy.ToPython(equal_nan);
  py := InvokeMethod('isclose', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function  TNDarrayGeneral.array_equal(a1: TNDarray): Boolean;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple([a1]);
  kwargs := TPyDict.Create;
  py := InvokeMethod('array_equal', pyargs, kwargs);
  Result := TNumPy.ToCsharp<Boolean>(py);
end;

Function  TNDarrayGeneral.array_equiv(a1: TNDarray): Boolean;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple([a1]);
  kwargs := TPyDict.Create;
  py := InvokeMethod('array_equiv', pyargs, kwargs);
  Result := TNumPy.ToCsharp<Boolean>(py);
end;

Function TNDarrayGeneral.greater(x1: TNDarray; var _out  :  TNDarray ; var where  :  TNDarray):TNDarray ;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple([x1]);
  kwargs := TPyDict.Create;
  if (_out <> nil) then kwargs['out'] := TNumPy.ToPython(_out);
  if (where <> nil) then kwargs['where'] := TNumPy.ToPython(where);
  py := InvokeMethod('greater', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function  TNDarrayGeneral.greater_equal(x1: TNDarray; var _out  :  TNDarray ; var where  :  TNDarray): TNDArray<Boolean>;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple([x1]);
  kwargs := TPyDict.Create;
  if (_out <> nil) then kwargs['out'] := TNumPy.ToPython(_out);
  if (where <> nil) then kwargs['where'] := TNumPy.ToPython(where);
  py := InvokeMethod('greater_equal', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDArray<Boolean>>(py);
end;

Function TNDarrayGeneral.less(x1: TNDarray; var _out  :  TNDarray ; var where  :  TNDarray):TNDarray ;

var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple([x1]);
  kwargs := TPyDict.Create;
  if (_out <> nil) then kwargs['out'] := TNumPy.ToPython(_out);
  if (where <> nil) then kwargs['where'] := TNumPy.ToPython(where);
  py := InvokeMethod('less', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNDarrayGeneral.less_equal(x1: TNDarray; var _out  :  TNDarray ; var where  :  TNDarray ):TNDarray ;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple([x1]);
  kwargs := TPyDict.Create;
  if (_out <> nil) then kwargs['out'] := TNumPy.ToPython(_out);
  if (where <> nil) then kwargs['where'] := TNumPy.ToPython(where);
  py := InvokeMethod('less_equal', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNDarrayGeneral.equal(x1: TNDarray; var _out  :  TNDarray  ; var where  :  TNDarray ):TNDarray ;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple([x1]);
  kwargs := TPyDict.Create;
  if (_out <> nil) then kwargs['out'] := TNumPy.ToPython(_out);
  if (where <> nil) then kwargs['where'] := TNumPy.ToPython(where);
  py := InvokeMethod('equal', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNDarrayGeneral.not_equal(x1: TNDarray; var _out  :  TNDarray  ; var where  :  TNDarray ):TNDarray ;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple([x1]);
  kwargs := TPyDict.Create;
  if (_out <> nil) then kwargs['out'] := TNumPy.ToPython(_out);
  if (where <> nil) then kwargs['where'] := TNumPy.ToPython(where);
  py := InvokeMethod('not_equal', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNDarrayGeneral.sin(var _out  :  TNDarray  ; var where  :  TNDarray):TNDarray ;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple();
  kwargs := TPyDict.Create;
  if (_out <> nil) then kwargs['out'] := TNumPy.ToPython(_out);
  if (where <> nil) then kwargs['where'] := TNumPy.ToPython(where);
  py := InvokeMethod('sin', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNDarrayGeneral.cos(var _out  :  TNDarray ; var where  :  TNDarray ):TNDarray ;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple();
  kwargs := TPyDict.Create;
  if (_out <> nil) then kwargs['out'] := TNumPy.ToPython(_out);
  if (where <> nil) then kwargs['where'] := TNumPy.ToPython(where);
  py := InvokeMethod('cos', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNDarrayGeneral.tan(var _out  :  TNDarray ; var where  :  TNDarray):TNDarray ;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple();
  kwargs := TPyDict.Create;
  if (_out <> nil) then kwargs['out'] := TNumPy.ToPython(_out);
  if (where <> nil) then kwargs['where'] := TNumPy.ToPython(where);
  py := InvokeMethod('tan', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNDarrayGeneral.arcsin(var _out  :  TNDarray ; var where  :  TNDarray):TNDarray ;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple();
  kwargs := TPyDict.Create;
  if (_out <> nil) then kwargs['out'] := TNumPy.ToPython(_out);
  if (where <> nil) then kwargs['where'] := TNumPy.ToPython(where);
  py := InvokeMethod('arcsin', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNDarrayGeneral.arccos(var _out  :  TNDarray ; var where  :  TNDarray):TNDarray ;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple();
  kwargs := TPyDict.Create;
  if (_out <> nil) then kwargs['out'] := TNumPy.ToPython(_out);
  if (where <> nil) then kwargs['where'] := TNumPy.ToPython(where);
  py := InvokeMethod('arccos', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNDarrayGeneral.arctan(var _out  :  TNDarray; var where  :  TNDarray):TNDarray ;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple();
  kwargs := TPyDict.Create;
  if (_out <> nil) then kwargs['out'] := TNumPy.ToPython(_out);
  if (where <> nil) then kwargs['where'] := TNumPy.ToPython(where);
  py := InvokeMethod('arctan', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNDarrayGeneral.hypot(x1: TNDarray; var _out  :  TNDarray; var where  :  TNDarray):TNDarray ;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple([x1]);
  kwargs := TPyDict.Create;
  if (_out <> nil) then kwargs['out'] := TNumPy.ToPython(_out);
  if (where <> nil) then kwargs['where'] := TNumPy.ToPython(where);
  py := InvokeMethod('hypot', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNDarrayGeneral.arctan2(x2: TNDarray; var _out  :  TNDarray; var where  :  TNDarray):TNDarray ;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple([x2]);
  kwargs := TPyDict.Create;
  if (_out <> nil) then kwargs['out'] := TNumPy.ToPython(_out);
  if (where <> nil) then kwargs['where'] := TNumPy.ToPython(where);
  py := InvokeMethod('arctan2', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNDarrayGeneral.degrees(var _out  :  TNDarray ; var where  :  TNDarray):TNDarray ;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple();
  kwargs := TPyDict.Create;
  if (_out <> nil) then kwargs['out'] := TNumPy.ToPython(_out);
  if (where <> nil) then kwargs['where'] := TNumPy.ToPython(where);
  py := InvokeMethod('degrees', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNDarrayGeneral.radians(var _out  :  TNDarray; var where  :  TNDarray):TNDarray ;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple();
  kwargs := TPyDict.Create;
  if (_out <> nil) then kwargs['out'] := TNumPy.ToPython(_out);
  if (where <> nil) then kwargs['where'] := TNumPy.ToPython(where);
  py := InvokeMethod('radians', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNDarrayGeneral.unwrap(discont : Double = 3.141592653589793; axis : Integer = -1):TNDarray;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple();
  kwargs := TPyDict.Create;
  if (discont<>3.141592653589793) then kwargs['discont'] := TNumPy.ToPython(discont);
  if (axis<>-1) then kwargs['axis'] := TNumPy.ToPython(axis);
  py := InvokeMethod('unwrap', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNDarrayGeneral.deg2rad(var _out  :  TNDarray ; var where  :  TNDarray):TNDarray ;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple();
  kwargs := TPyDict.Create;
  if (_out <> nil) then kwargs['out'] := TNumPy.ToPython(_out);
  if (where <> nil) then kwargs['where'] := TNumPy.ToPython(where);
  py := InvokeMethod('deg2rad', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNDarrayGeneral.rad2deg(var _out  :  TNDarray ; var where  :  TNDarray ):TNDarray ;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple();
  kwargs := TPyDict.Create;
  if (_out <> nil) then kwargs['out'] := TNumPy.ToPython(_out);
  if (where <> nil) then kwargs['where'] := TNumPy.ToPython(where);
  py := InvokeMethod('rad2deg', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNDarrayGeneral.sinh(var _out  :  TNDarray; var where  :  TNDarray):TNDarray ;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple();
  kwargs := TPyDict.Create;
  if (_out <> nil) then kwargs['out'] := TNumPy.ToPython(_out);
  if (where <> nil) then kwargs['where'] := TNumPy.ToPython(where);
  py := InvokeMethod('sinh', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNDarrayGeneral.cosh(var _out  :  TNDarray; var where  :  TNDarray ):TNDarray ;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple();
  kwargs := TPyDict.Create;
  if (_out <> nil) then kwargs['out'] := TNumPy.ToPython(_out);
  if (where <> nil) then kwargs['where'] := TNumPy.ToPython(where);
  py := InvokeMethod('cosh', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNDarrayGeneral.tanh(var _out  :  TNDarray ; var where  :  TNDarray ):TNDarray ;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple();
  kwargs := TPyDict.Create;
  if (_out <> nil) then kwargs['out'] := TNumPy.ToPython(_out);
  if (where <> nil) then kwargs['where'] := TNumPy.ToPython(where);
  py := InvokeMethod('tanh', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNDarrayGeneral.arcsinh(var _out  :  TNDarray ; var where  :  TNDarray ):TNDarray ;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple();
  kwargs := TPyDict.Create;
  if (_out <> nil) then kwargs['out'] := TNumPy.ToPython(_out);
  if (where <> nil) then kwargs['where'] := TNumPy.ToPython(where);
  py := InvokeMethod('arcsinh', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNDarrayGeneral.arccosh(var _out  :  TNDarray ; var where  :  TNDarray ):TNDarray ;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple();
  kwargs := TPyDict.Create;
  if (_out <> nil) then kwargs['out'] := TNumPy.ToPython(_out);
  if (where <> nil) then kwargs['where'] := TNumPy.ToPython(where);
  py := InvokeMethod('arccosh', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNDarrayGeneral.arctanh(var _out  :  TNDarray ; var where  :  TNDarray ):TNDarray ;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple();
  kwargs := TPyDict.Create;
  if (_out <> nil) then kwargs['out'] := TNumPy.ToPython(_out);
  if (where <> nil) then kwargs['where'] := TNumPy.ToPython(where);
  py := InvokeMethod('arctanh', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNDarrayGeneral.around(decimals : Integer; var _out  :  TNDarray ):TNDarray ;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple();
  kwargs := TPyDict.Create;
  if (decimals<>0) then kwargs['decimals'] := TNumPy.ToPython(decimals);
  if (_out <> nil) then kwargs['out'] := TNumPy.ToPython(_out);
  py := InvokeMethod('around', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNDarrayGeneral.rint(var _out  :  TNDarray; var where  :  TNDarray ):TNDarray ;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple();
  kwargs := TPyDict.Create;
  if (_out <> nil) then kwargs['out'] := TNumPy.ToPython(_out);
  if (where <> nil) then kwargs['where'] := TNumPy.ToPython(where);
  py := InvokeMethod('rint', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNDarrayGeneral.fix(y  :  TNDarray  = nil):TNDarray ;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple();
  kwargs := TPyDict.Create;
  if (y<>nil) then kwargs['y'] := TNumPy.ToPython(y);
  py := InvokeMethod('fix', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNDarrayGeneral.floor(var _out  :  TNDarray ; var where  :  TNDarray):TNDarray ;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple();
  kwargs := TPyDict.Create;
  if (_out <> nil) then kwargs['out'] := TNumPy.ToPython(_out);
  if (where <> nil) then kwargs['where'] := TNumPy.ToPython(where);
  py := InvokeMethod('floor', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNDarrayGeneral.ceil(var _out  :  TNDarray ; var where  :  TNDarray ):TNDarray ;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple();
  kwargs := TPyDict.Create;
  if (_out <> nil) then kwargs['out'] := TNumPy.ToPython(_out);
  if (where <> nil) then kwargs['where'] := TNumPy.ToPython(where);
  py := InvokeMethod('ceil', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNDarrayGeneral.trunc(var _out  :  TNDarray ; var where  :  TNDarray  ):TNDarray ;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple();
  kwargs := TPyDict.Create;
  if (_out <> nil) then kwargs['out'] := TNumPy.ToPython(_out);
  if (where <> nil) then kwargs['where'] := TNumPy.ToPython(where);
  py := InvokeMethod('trunc', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNDarrayGeneral.prod(axis: TArray<Integer>; dtype  : TDtype  ; var _out  :  TNDarray ; keepdims  : PBoolean  ; initial  : TValue  ):TNDarray ;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple();
  kwargs := TPyDict.Create;
  if (axis<>nil) then kwargs['axis'] := TNumPy.ToPython(TValue.FromArray<Integer>(axis));
  if (dtype<>nil) then kwargs['dtype'] := TNumPy.ToPython(dtype);
  if (_out <> nil) then kwargs['out'] := TNumPy.ToPython(_out);
  if (keepdims<>nil) then kwargs['keepdims'] := TNumPy.ToPython(keepdims^);

  { TODO -oMax -c : verificare 06/02/2020 21:08:15 }
  if (not initial.IsEmpty) then kwargs['initial'] := TNumPy.ToPython(initial);
  py := InvokeMethod('prod', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNDarrayGeneral.sum(axis : TArray<Integer>; dtype  : TDtype ; var _out  :  TNDarray ; keepdims  : PBoolean ; initial  : TValue ):TNDarray ;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple();
  kwargs := TPyDict.Create;
  if (Length(axis) > 0) then kwargs['axis'] := TNumPy.ToPython(TValue.FromArray<Integer>(axis));
  if (dtype<>nil) then kwargs['dtype'] := TNumPy.ToPython(dtype);
  if (_out <> nil) then kwargs['out'] := TNumPy.ToPython(_out);
  if (keepdims<>nil) then kwargs['keepdims'] := TNumPy.ToPython(keepdims^);
  if (not initial.IsEmpty) then kwargs['initial'] := TNumPy.ToPython(initial);
  py := InvokeMethod('sum', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNDarrayGeneral.sum:TNDarray ;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple();
  kwargs := TPyDict.Create;

  py := InvokeMethod('sum', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNDarrayGeneral.nanprod(axis : TArray<Integer>; dtype  : TDtype  ; var _out  :  TNDarray; keepdims  : PBoolean):TNDarray ;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple();
  kwargs := TPyDict.Create;
  if (Length(axis) > 0) then kwargs['axis'] := TNumPy.ToPython(TValue.FromArray<Integer>(axis));
  if (dtype<>nil) then kwargs['dtype'] := TNumPy.ToPython(dtype);
  if (_out <> nil) then kwargs['out'] := TNumPy.ToPython(_out);
  if (keepdims<>nil) then kwargs['keepdims'] := TNumPy.ToPython(keepdims^);
  py := InvokeMethod('nanprod', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNDarrayGeneral.nansum(axis : TArray<Integer>; dtype  : TDtype; var _out  :  TNDarray ; keepdims  : PBoolean):TNDarray ;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple();
  kwargs := TPyDict.Create;
  if (axis<>nil) then kwargs['axis'] := TNumPy.ToPython(TValue.FromArray<Integer>(axis));
  if (dtype<>nil) then kwargs['dtype'] := TNumPy.ToPython(dtype);
  if (_out <> nil) then kwargs['out'] := TNumPy.ToPython(_out);
  if (keepdims<>nil) then kwargs['keepdims'] := TNumPy.ToPython(keepdims^);
  py := InvokeMethod('nansum', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNDarrayGeneral.cumprod(axis  : PInteger ; dtype  : TDtype ; var _out  :  TNDarray):TNDarray ;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple();
  kwargs := TPyDict.Create;
  if (axis<>nil) then kwargs['axis'] := TNumPy.ToPython(axis^);
  if (dtype<>nil) then kwargs['dtype'] := TNumPy.ToPython(dtype);
  if (_out <> nil) then kwargs['out'] := TNumPy.ToPython(_out);
  py := InvokeMethod('cumprod', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNDarrayGeneral.cumsum(axis  : PInteger; dtype  : TDtype ; var _out  :  TNDarray):TNDarray ;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple();
  kwargs := TPyDict.Create;
  if (axis<>nil) then kwargs['axis'] := TNumPy.ToPython(axis^);
  if (dtype<>nil) then kwargs['dtype'] := TNumPy.ToPython(dtype);
  if (_out <> nil) then kwargs['out'] := TNumPy.ToPython(_out);
  py := InvokeMethod('cumsum', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNDarrayGeneral.nancumprod(axis  : PInteger ; dtype  : TDtype ; var _out  :  TNDarray ):TNDarray ;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple();
  kwargs := TPyDict.Create;
  if (axis<>nil) then kwargs['axis'] := TNumPy.ToPython(axis^);
  if (dtype<>nil) then kwargs['dtype'] := TNumPy.ToPython(dtype);
  if (_out <> nil) then kwargs['out'] := TNumPy.ToPython(_out);
  py := InvokeMethod('nancumprod', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNDarrayGeneral.nancumsum(axis  : PInteger; dtype  : TDtype; var _out  :  TNDarray):TNDarray ;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple();
  kwargs := TPyDict.Create;
  if (axis<>nil) then kwargs['axis'] := TNumPy.ToPython(axis^);
  if (dtype<>nil) then kwargs['dtype'] := TNumPy.ToPython(dtype);
  if (_out <> nil) then kwargs['out'] := TNumPy.ToPython(_out);
  py := InvokeMethod('nancumsum', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNDarrayGeneral.diff(n : Integer = 1; axis : Integer = -1; append  :  TNDarray  = nil; prepend  :  TNDarray  = nil):TNDarray ;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple();
  kwargs := TPyDict.Create;
  if (n<>1) then kwargs['n'] := TNumPy.ToPython(n);
  if (axis<>-1) then kwargs['axis'] := TNumPy.ToPython(axis);
  if (append<>nil) then kwargs['append'] := TNumPy.ToPython(append);
  if (prepend<>nil) then kwargs['prepend'] := TNumPy.ToPython(prepend);
  py := InvokeMethod('diff', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNDarrayGeneral.ediff1d(to_end : TNDarray = nil; to_begin : TNDarray = nil):TNDarray;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple();
  kwargs := TPyDict.Create;
  if (to_end<>nil) then kwargs['to_end'] := TNumPy.ToPython(to_end);
  if (to_begin<>nil) then kwargs['to_var'] := TNumPy.ToPython(to_begin);
  py := InvokeMethod('ediff1d', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNDarrayGeneral.gradient(_varargs  :  TNDarray ; edge_order : PInteger ; axis : TArray<Integer>) :TNDarray;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple();
  kwargs := TPyDict.Create;
  if (_varargs <> nil) then kwargs['varargs'] := TNumPy.ToPython(_varargs);
  if (edge_order<>nil) then kwargs['edge_order'] := TNumPy.ToPython(edge_order^);
  if (Length(axis) > 0) then kwargs['axis'] := TNumPy.ToPython(TValue.FromArray<Integer>(axis));
  py := InvokeMethod('gradient', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNDarrayGeneral.cross(b :  TNDarray ; axisa : Integer = -1; axisb : Integer = -1; axisc : Integer = -1; axis  : PInteger = nil):TNDarray ;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple([b]);
  kwargs := TPyDict.Create;
  if (axisa<>-1) then kwargs['axisa'] := TNumPy.ToPython(axisa);
  if (axisb<>-1) then kwargs['axisb'] := TNumPy.ToPython(axisb);
  if (axisc<>-1) then kwargs['axisc'] := TNumPy.ToPython(axisc);
  if (axis<>nil) then kwargs['axis'] := TNumPy.ToPython(axis^);
  py := InvokeMethod('cross', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function  TNDarrayGeneral.trapz(x  :  TNDarray  = nil; dx : Double = 1.0; axis : Integer = -1): Double;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple();
  kwargs := TPyDict.Create;
  if (x<>nil) then kwargs['x'] := TNumPy.ToPython(x);
  if (dx<>1.0) then kwargs['dx'] := TNumPy.ToPython(dx);
  if (axis<>-1) then kwargs['axis'] := TNumPy.ToPython(axis);
  py := InvokeMethod('trapz', pyargs, kwargs);
  Result := TNumPy.ToCsharp<Double>(py);
end;

Function TNDarrayGeneral.exp(var _out  :  TNDarray ; var where  :  TNDarray):TNDarray ;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple();
  kwargs := TPyDict.Create;
  if (_out <> nil) then kwargs['out'] := TNumPy.ToPython(_out);
  if (where <> nil) then kwargs['where'] := TNumPy.ToPython(where);
  py := InvokeMethod('exp', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNDarrayGeneral.expm1(var _out  :  TNDarray ; var where  :  TNDarray):TNDarray ;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple();
  kwargs := TPyDict.Create;
  if (_out <> nil) then kwargs['out'] := TNumPy.ToPython(_out);
  if (where <> nil) then kwargs['where'] := TNumPy.ToPython(where);
  py := InvokeMethod('expm1', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNDarrayGeneral.exp2(var _out  :  TNDarray ; var where  :  TNDarray ):TNDarray ;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple();
  kwargs := TPyDict.Create;
  if (_out <> nil) then kwargs['out'] := TNumPy.ToPython(_out);
  if (where <> nil) then kwargs['where'] := TNumPy.ToPython(where);
  py := InvokeMethod('exp2', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNDarrayGeneral.log(var _out  :  TNDarray ; var where  :  TNDarray ):TNDarray ;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple();
  kwargs := TPyDict.Create;
  if (_out <> nil) then kwargs['out'] := TNumPy.ToPython(_out);
  if (where <> nil) then kwargs['where'] := TNumPy.ToPython(where);
  py := InvokeMethod('log', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNDarrayGeneral.log10(var _out  :  TNDarray ; var where  :  TNDarray ):TNDarray ;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple();
  kwargs := TPyDict.Create;
  if (_out <> nil) then kwargs['out'] := TNumPy.ToPython(_out);
  if (where <> nil) then kwargs['where'] := TNumPy.ToPython(where);
  py := InvokeMethod('log10', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNDarrayGeneral.log2(var _out  :  TNDarray ; var where  :  TNDarray ):TNDarray ;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple();
  kwargs := TPyDict.Create;
  if (_out <> nil) then kwargs['out'] := TNumPy.ToPython(_out);
  if (where <> nil) then kwargs['where'] := TNumPy.ToPython(where);
  py := InvokeMethod('log2', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNDarrayGeneral.log1p(var _out  :  TNDarray ; var where  :  TNDarray ):TNDarray ;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple();
  kwargs := TPyDict.Create;
  if (_out <> nil) then kwargs['out'] := TNumPy.ToPython(_out);
  if (where <> nil) then kwargs['where'] := TNumPy.ToPython(where);
  py := InvokeMethod('log1p', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNDarrayGeneral.logaddexp(x1: TNDarray; var _out  :  TNDarray ; var where  :  TNDarray ):TNDarray ;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple([x1]);
  kwargs := TPyDict.Create;
  if (_out <> nil) then kwargs['out'] := TNumPy.ToPython(_out);
  if (where <> nil) then kwargs['where'] := TNumPy.ToPython(where);
  py := InvokeMethod('logaddexp', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNDarrayGeneral.logaddexp2(x1: TNDarray; var _out  :  TNDarray ; var where  :  TNDarray ):TNDarray ;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple([x1]);
  kwargs := TPyDict.Create;
  if (_out <> nil) then kwargs['out'] := TNumPy.ToPython(_out);
  if (where <> nil) then kwargs['where'] := TNumPy.ToPython(where);
  py := InvokeMethod('logaddexp2', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNDarrayGeneral.sinc:TNDarray ;
var
   py     : TPythonObject;
begin
  py := InvokeMethod('sinc',[]);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNDarrayGeneral.signbit(var _out  :  TNDarray ; var where  :  TNDarray ):TNDarray ;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple();
  kwargs := TPyDict.Create;
  if (_out <> nil) then kwargs['out'] := TNumPy.ToPython(_out);
  if (where <> nil) then kwargs['where'] := TNumPy.ToPython(where);
  py := InvokeMethod('signbit', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNDarrayGeneral.copysign(x2 : TNDarray; var _out  :  TNDarray ; var where  :  TNDarray ):TNDarray ;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple([x2]);
  kwargs := TPyDict.Create;
  if (_out <> nil) then kwargs['out'] := TNumPy.ToPython(_out);
  if (where <> nil) then kwargs['where'] := TNumPy.ToPython(where);
  py := InvokeMethod('copysign', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;


 { TODO -oMax -c :  verificare parametri in uscita se ok con array 06/02/2020 22:32:32 }
function TNDarrayGeneral.frexp(out1 : TNDarray ;  out2 : TNDarray ; var _out  :  TNDarray ; var where  :  TNDarray ): TArray<TNDarray>;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
   res    : TArray<TPythonObject>;
begin
  pyargs := TNumPy.ToTuple();
  kwargs := TPyDict.Create;
  if (out1 <> nil) then kwargs['out1'] := TNumPy.ToPython(out1);
  if (out2 <> nil) then kwargs['out2'] := TNumPy.ToPython(out2);
  if (_out <> nil) then kwargs['out'] := TNumPy.ToPython(_out);
  if (where <> nil) then kwargs['where'] := TNumPy.ToPython(where);
  py := InvokeMethod('frexp', pyargs, kwargs);

  res := py.AsArrayofPyObj;

  SetLength(Result,2);
  Result[0] := TNumPy.ToCsharp<TNDarray>(res[0]);
  Result[1] := TNumPy.ToCsharp<TNDarray>(res[1]);

end;

Function TNDarrayGeneral.ldexp(x2: TNDarray; var _out  :  TNDarray  ; var where  :  TNDarray ):TNDarray ;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple([x2]);
  kwargs := TPyDict.Create;
  if (_out <> nil) then kwargs['out'] := TNumPy.ToPython(_out);
  if (where <> nil) then kwargs['where'] := TNumPy.ToPython(where);
  py := InvokeMethod('ldexp', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNDarrayGeneral.nextafter(x2: TNDarray; var _out  :  TNDarray ; var where  :  TNDarray):TNDarray ;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple([x2]);
  kwargs := TPyDict.Create;
  if (_out <> nil) then kwargs['out'] := TNumPy.ToPython(_out);
  if (where <> nil) then kwargs['where'] := TNumPy.ToPython(where);
  py := InvokeMethod('nextafter', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNDarrayGeneral.spacing(var _out  :  TNDarray  ; var where  :  TNDarray ):TNDarray ;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple();
  kwargs := TPyDict.Create;
  if (_out <> nil) then kwargs['out'] := TNumPy.ToPython(_out);
  if (where <> nil) then kwargs['where'] := TNumPy.ToPython(where);
  py := InvokeMethod('spacing', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNDarrayGeneral.lcm(x1: TNDarray):TNDarray ;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple([x1]);
  kwargs := TPyDict.Create;
  py := InvokeMethod('lcm', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNDarrayGeneral.gcd(x1: TNDarray):TNDarray ;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple([x1]);
  kwargs := TPyDict.Create;
  py := InvokeMethod('gcd', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNDarrayGeneral.add(x1: TNDarray; var _out  :  TNDarray  ; var where  :  TNDarray  ):TNDarray ;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple([x1]);
  kwargs := TPyDict.Create;
  if (_out <> nil) then kwargs['out'] := TNumPy.ToPython(_out);
  if (where <> nil) then kwargs['where'] := TNumPy.ToPython(where);
  py := InvokeMethod('add', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNDarrayGeneral.reciprocal(var _out  :  TNDarray  ; var where  :  TNDarray ):TNDarray ;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple();
  kwargs := TPyDict.Create;
  if (_out <> nil) then kwargs['out'] := TNumPy.ToPython(_out);
  if (where <> nil) then kwargs['where'] := TNumPy.ToPython(where);
  py := InvokeMethod('reciprocal', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNDarrayGeneral.positive:TNDarray ;
var
   py     : TPythonObject;
begin
  py := InvokeMethod('positive',[]);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNDarrayGeneral.negative(var _out  :  TNDarray  ; var where  :  TNDarray  ):TNDarray ;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple();
  kwargs := TPyDict.Create;
  if (_out <> nil) then kwargs['out'] := TNumPy.ToPython(_out);
  if (where <> nil) then kwargs['where'] := TNumPy.ToPython(where);
  py := InvokeMethod('negative', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNDarrayGeneral.multiply(x1: TNDarray; var _out  :  TNDarray ; var where  :  TNDarray ):TNDarray ;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple([x1]);
  kwargs := TPyDict.Create;
  if (_out <> nil) then kwargs['out'] := TNumPy.ToPython(_out);
  if (where <> nil) then kwargs['where'] := TNumPy.ToPython(where);
  py := InvokeMethod('multiply', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNDarrayGeneral.divide(x2: TNDarray; var _out  :  TNDarray  ; var where  :  TNDarray ):TNDarray ;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple([x2]);
  kwargs := TPyDict.Create;
  if (_out <> nil) then kwargs['out'] := TNumPy.ToPython(_out);
  if (where <> nil) then kwargs['where'] := TNumPy.ToPython(where);
  py := InvokeMethod('divide', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNDarrayGeneral.power(x2: TNDarray; var _out  :  TNDarray  ; var where  :  TNDarray ):TNDarray ;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple([x2]);
  kwargs := TPyDict.Create;
  if (_out <> nil) then kwargs['out'] := TNumPy.ToPython(_out);
  if (where <> nil) then kwargs['where'] := TNumPy.ToPython(where);
  py := InvokeMethod('power', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNDarrayGeneral.subtract(x1: TNDarray; var _out  :  TNDarray ; var where  :  TNDarray):TNDarray ;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple([x1]);
  kwargs := TPyDict.Create;
  if (_out <> nil) then kwargs['out'] := TNumPy.ToPython(_out);
  if (where <> nil) then kwargs['where'] := TNumPy.ToPython(where);
  py := InvokeMethod('subtract', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNDarrayGeneral.true_divide(x2: TNDarray; var _out  :  TNDarray ; var where  :  TNDarray ):TNDarray ;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple([x2]);
  kwargs := TPyDict.Create;
  if (_out <> nil) then kwargs['out'] := TNumPy.ToPython(_out);
  if (where <> nil) then kwargs['where'] := TNumPy.ToPython(where);
  py := InvokeMethod('true_divide', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNDarrayGeneral.floor_divide(x2: TNDarray; var _out  :  TNDarray ; var where  :  TNDarray ):TNDarray ;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple([x2]);
  kwargs := TPyDict.Create;
  if (_out <> nil) then kwargs['out'] := TNumPy.ToPython(_out);
  if (where <> nil) then kwargs['where'] := TNumPy.ToPython(where);
  py := InvokeMethod('floor_divide', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNDarrayGeneral.float_power(x2: TNDarray; var _out  :  TNDarray  ; var where  :  TNDarray ):TNDarray ;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple([x2]);
  kwargs := TPyDict.Create;
  if (_out <> nil) then kwargs['out'] := TNumPy.ToPython(_out);
  if (where <> nil) then kwargs['where'] := TNumPy.ToPython(where);
  py := InvokeMethod('float_power', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNDarrayGeneral.fmod(x2: TNDarray; var _out  :  TNDarray ; var where  :  TNDarray ):TNDarray ;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple([x2]);
  kwargs := TPyDict.Create;
  if (_out <> nil) then kwargs['out'] := TNumPy.ToPython(_out);
  if (where <> nil) then kwargs['where'] := TNumPy.ToPython(where);
  py := InvokeMethod('fmod', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNDarrayGeneral.&mod(x2: TNDarray; var _out  :  TNDarray  ; var where  :  TNDarray  ):TNDarray ;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple([x2]);
  kwargs := TPyDict.Create;
  if (_out <> nil) then kwargs['out'] := TNumPy.ToPython(_out);
  if (where <> nil) then kwargs['where'] := TNumPy.ToPython(where);
  py := InvokeMethod('mod', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function  TNDarrayGeneral.modf(var _out  :  TNDarray  ; var where  :  TNDarray  ):TArray<TNDarray>;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
   res    : TArray<TPythonObject>;
begin
  pyargs := TNumPy.ToTuple();
  kwargs := TPyDict.Create;
  if (_out <> nil) then kwargs['out'] := TNumPy.ToPython(_out);
  if (where <> nil) then kwargs['where'] := TNumPy.ToPython(where);
  py := InvokeMethod('modf', pyargs, kwargs);

  res := py.AsArrayofPyObj;

  SetLength(Result,2);
  Result[0] := TNumPy.ToCsharp<TNDarray>(res[0]);
  Result[1] := TNumPy.ToCsharp<TNDarray>(res[1]);

end;

Function TNDarrayGeneral.remainder(x2: TNDarray; var _out  :  TNDarray  ; var where  :  TNDarray  ):TNDarray ;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple([x2]);
  kwargs := TPyDict.Create;
  if (_out <> nil) then kwargs['out'] := TNumPy.ToPython(_out);
  if (where <> nil) then kwargs['where'] := TNumPy.ToPython(where);
  py := InvokeMethod('remainder', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNDarrayGeneral.divmod(x2: TNDarray; var _out  :  TNDarray  ; var where  :  TNDarray  ):TArray<TNDarray>;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
   res    : TArray<TPythonObject>;
begin
  pyargs := TNumPy.ToTuple([x2]);
  kwargs := TPyDict.Create;
  if (_out <> nil) then kwargs['out'] := TNumPy.ToPython(_out);
  if (where <> nil) then kwargs['where'] := TNumPy.ToPython(where);
  py := InvokeMethod('divmod', pyargs, kwargs);

  res := py.AsArrayofPyObj;

  SetLength(Result,2);
  Result[0] := TNumPy.ToCsharp<TNDarray>(res[0]);
  Result[1] := TNumPy.ToCsharp<TNDarray>(res[1]);

end;

Function TNDarrayGeneral.angle(deg : Boolean = false):TNDarray ;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple();
  kwargs := TPyDict.Create;
  if (deg<>false) then kwargs['deg'] := TNumPy.ToPython(deg);
  py := InvokeMethod('angle', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNDarrayGeneral.real:TNDarray ;
var
   py     : TPythonObject;
begin
  py := InvokeMethod('real',[]);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNDarrayGeneral.imag:TNDarray ;
var
   py     : TPythonObject;
begin
  py := InvokeMethod('imag',[]);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNDarrayGeneral.conj(var _out  :  TNDarray  ; var where  :  TNDarray  ):TNDarray ;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple();
  kwargs := TPyDict.Create;
  if (_out <> nil) then kwargs['out'] := TNumPy.ToPython(_out);
  if (where <> nil) then kwargs['where'] := TNumPy.ToPython(where);
  py := InvokeMethod('conj', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNDarrayGeneral.convolve(v :  TNDarray ; mode : string = 'full'):TNDarray ;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple([v]);
  kwargs := TPyDict.Create;
  if (mode<>'full') then kwargs['mode'] := TNumPy.ToPython(mode);
  py := InvokeMethod('convolve', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNDarrayGeneral.clip(a_min: TNDarray;  a_max: TNDarray; var _out  :  TNDarray ):TNDarray;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple([a_min,a_max]);
  kwargs := TPyDict.Create;
  if (_out <> nil) then kwargs['out'] := TNumPy.ToPython(_out);
  py := InvokeMethod('clip', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNDarrayGeneral.sqrt(var _out  :  TNDarray ; var where  :  TNDarray):TNDarray ;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple();
  kwargs := TPyDict.Create;
  if (_out <> nil) then kwargs['out'] := TNumPy.ToPython(_out);
  if (where <> nil) then kwargs['where'] := TNumPy.ToPython(where);
  py := InvokeMethod('sqrt', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNDarrayGeneral.cbrt(var _out  :  TNDarray  ; var where  :  TNDarray  ):TNDarray ;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple();
  kwargs := TPyDict.Create;
  if (_out <> nil) then kwargs['out'] := TNumPy.ToPython(_out);
  if (where <> nil) then kwargs['where'] := TNumPy.ToPython(where);
  py := InvokeMethod('cbrt', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNDarrayGeneral.square(var _out  :  TNDarray ; var where  :  TNDarray  ):TNDarray ;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple();
  kwargs := TPyDict.Create;
  if (_out <> nil) then kwargs['out'] := TNumPy.ToPython(_out);
  if (where <> nil) then kwargs['where'] := TNumPy.ToPython(where);
  py := InvokeMethod('square', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNDarrayGeneral.absolute(var _out  :  TNDarray  ; var where  :  TNDarray  ):TNDarray ;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple();
  kwargs := TPyDict.Create;
  if (_out <> nil) then kwargs['out'] := TNumPy.ToPython(_out);
  if (where <> nil) then kwargs['where'] := TNumPy.ToPython(where);
  py := InvokeMethod('absolute', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNDarrayGeneral.fabs(var _out  :  TNDarray ; var where  :  TNDarray ):TNDarray ;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple();
  kwargs := TPyDict.Create;
  if (_out <> nil) then kwargs['out'] := TNumPy.ToPython(_out);
  if (where <> nil) then kwargs['where'] := TNumPy.ToPython(where);
  py := InvokeMethod('fabs', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNDarrayGeneral.sign(var _out  :  TNDarray  ; var where  :  TNDarray  ):TNDarray ;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple();
  kwargs := TPyDict.Create;
  if (_out <> nil) then kwargs['out'] := TNumPy.ToPython(_out);
  if (where <> nil) then kwargs['where'] := TNumPy.ToPython(where);
  py := InvokeMethod('sign', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNDarrayGeneral.heaviside(x2: TNDarray; var _out  :  TNDarray  ; var where  :  TNDarray ):TNDarray ;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple([x2]);
  kwargs := TPyDict.Create;
  if (_out <> nil) then kwargs['out'] := TNumPy.ToPython(_out);
  if (where <> nil) then kwargs['where'] := TNumPy.ToPython(where);
  py := InvokeMethod('heaviside', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNDarrayGeneral.maximum(x1: TNDarray; var _out  :  TNDarray ; var where  :  TNDarray ):TNDarray ;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple([x1]);
  kwargs := TPyDict.Create;
  if (_out <> nil) then kwargs['out'] := TNumPy.ToPython(_out);
  if (where <> nil) then kwargs['where'] := TNumPy.ToPython(where);
  py := InvokeMethod('maximum', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNDarrayGeneral.minimum(x1: TNDarray; var _out  :  TNDarray  ; var where  :  TNDarray  ):TNDarray ;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple([x1]);
  kwargs := TPyDict.Create;
  if (_out <> nil) then kwargs['out'] := TNumPy.ToPython(_out);
  if (where <> nil) then kwargs['where'] := TNumPy.ToPython(where);
  py := InvokeMethod('minimum', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNDarrayGeneral.fmax(x1: TNDarray; var _out  :  TNDarray  ; var where  :  TNDarray  ):TNDarray ;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple([x1]);
  kwargs := TPyDict.Create;
  if (_out <> nil) then kwargs['out'] := TNumPy.ToPython(_out);
  if (where <> nil) then kwargs['where'] := TNumPy.ToPython(where);
  py := InvokeMethod('fmax', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNDarrayGeneral.fmin(x1: TNDarray; var _out  :  TNDarray  ; var where  :  TNDarray  ):TNDarray ;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple([x1]);
  kwargs := TPyDict.Create;
  if (_out <> nil) then kwargs['out'] := TNumPy.ToPython(_out);
  if (where <> nil) then kwargs['where'] := TNumPy.ToPython(where);
  py := InvokeMethod('fmin', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNDarrayGeneral.nan_to_num(copy : Boolean= true):TNDarray ;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple();
  kwargs := TPyDict.Create;
  if (copy<>true) then kwargs['copy'] := TNumPy.ToPython(copy);
  py := InvokeMethod('nan_to_num', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNDarrayGeneral.real_if_close(tol : Double = 100):TNDarray ;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple();
  kwargs := TPyDict.Create;
  if (tol<>100) then kwargs['tol'] := TNumPy.ToPython(tol);
  py := InvokeMethod('real_if_close', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

(*
Function  TNDarrayGeneral.float or complex (corresponding fp : to  ); or ndarray interp(1-D sequence of xp : floats ; 1-D sequence of float or fp : complex ; optional float or complex corresponding to left  : fp  = nil; optional float or complex corresponding to right  : fp  = nil; None or period  : float  = nil);
var
   np     : Tnp;
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple([xp,fp]);
  kwargs := TPyDict.Create;
  if (left<>null) then kwargs['left'] := TNumPy.ToPython(left);
  if (right<>null) then kwargs['right'] := TNumPy.ToPython(right);
  if (period<>null) then kwargs['period'] := TNumPy.ToPython(period);
  py := InvokeMethod('interp', pyargs, kwargs);
  Result := TNumPy.ToCsharp<float or complex (corresponding fp : to  ); or ndarray>(py);
end;
*)
Function TNDarrayGeneral.pad(pad_width: TNDarray; mode : string ; stat_length : TArray<Integer>;  constant_values : TArray<Integer>; end_values : TArray<Integer>; reflect_type : string = null):TNDarray;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple([pad_width,mode]);
  kwargs := TPyDict.Create;
  if (Length(stat_length) > 0) then kwargs['stat_length'] := TNumPy.ToPython(TValue.FromArray<Integer>(stat_length));
  if (Length(constant_values) > 0) then kwargs['constant_values'] := TNumPy.ToPython(TValue.FromArray<Integer>(constant_values));
  if (Length(end_values) >0) then kwargs['end_values'] := TNumPy.ToPython(TValue.FromArray<Integer>(end_values));
  if (reflect_type<>null) then kwargs['reflect_type'] := TNumPy.ToPython(reflect_type);
  py := InvokeMethod('pad', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNDarrayGeneral.in1d(ar2: TNDarray; assume_unique :Boolean = false; invert : Boolean = false): TNDarray;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple([ar2]);
  kwargs := TPyDict.Create;
  if (assume_unique<>false) then kwargs['assume_unique'] := TNumPy.ToPython(assume_unique);
  if (invert<>false) then kwargs['invert'] := TNumPy.ToPython(invert);
  py := InvokeMethod('in1d', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNDarrayGeneral.intersect1d(ar1: TNDarray; assume_unique : Boolean = false;  return_indices : Boolean = false):TArray<TNDarray>;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
   res    : TArray<TPythonObject>;
begin
  pyargs := TNumPy.ToTuple([ar1]);
  kwargs := TPyDict.Create;
  if (assume_unique<>false) then kwargs['assume_unique'] := TNumPy.ToPython(assume_unique);
  if (return_indices<>false) then kwargs['return_indices'] := TNumPy.ToPython(return_indices);
  py := InvokeMethod('intersect1d', pyargs, kwargs);

  res := py.AsArrayofPyObj;

  SetLength(Result,3);
  Result[0] := TNumPy.ToCsharp<TNDarray>(res[0]);
  Result[1] := TNumPy.ToCsharp<TNDarray>(res[1]);
  Result[2] := TNumPy.ToCsharp<TNDarray>(res[2]);

end;

Function TNDarrayGeneral.isin(test_elements: TNDarray; assume_unique : Boolean = false; invert : Boolean = false): TNDarray;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple([test_elements]);
  kwargs := TPyDict.Create;
  if (assume_unique<>false) then kwargs['assume_unique'] := TNumPy.ToPython(assume_unique);
  if (invert<>false) then kwargs['invert'] := TNumPy.ToPython(invert);
  py := InvokeMethod('isin', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNDarrayGeneral.setdiff1d(ar2: TNDarray; assume_unique : Boolean = false): TNDarray;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple([ar2]);
  kwargs := TPyDict.Create;
  if (assume_unique<>false) then kwargs['assume_unique'] := TNumPy.ToPython(assume_unique);
  py := InvokeMethod('setdiff1d', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNDarrayGeneral.setxor1d(ar1: TNDarray; assume_unique: Boolean = false):TNDarray;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple([ar1]);
  kwargs := TPyDict.Create;
  if (assume_unique<>false) then kwargs['assume_unique'] := TNumPy.ToPython(assume_unique);
  py := InvokeMethod('setxor1d', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNDarrayGeneral.union1d(ar1: TNDarray):TNDarray ;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple([ar1]);
  kwargs := TPyDict.Create;
  py := InvokeMethod('union1d', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNDarrayGeneral.sort(axis: Integer = -1; kind : string = 'quicksort'; order  : string  = null): TNDarray;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple();
  kwargs := TPyDict.Create;
  if (axis<>-1) then kwargs['axis'] := TNumPy.ToPython(axis);
  if (kind<>'quicksort') then kwargs['kind'] := TNumPy.ToPython(kind);
  if (order<>null) then kwargs['order'] := TNumPy.ToPython(order);
  py := InvokeMethod('sort', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNDarrayGeneral.lexsort(axis: Integer = -1):TNDarray ;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple();
  kwargs := TPyDict.Create;
  if (axis<>-1) then kwargs['axis'] := TNumPy.ToPython(axis);
  py := InvokeMethod('lexsort', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNDarrayGeneral.argsort(axis: integer = -1; kind : string = 'quicksort'; order  : string  = null):TNDarray ;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple();
  kwargs := TPyDict.Create;
  if (axis<>-1) then kwargs['axis'] := TNumPy.ToPython(axis);
  if (kind<>'quicksort') then kwargs['kind'] := TNumPy.ToPython(kind);
  if (order<>null) then kwargs['order'] := TNumPy.ToPython(order);
  py := InvokeMethod('argsort', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNDarrayGeneral.msort:TNDarray ;
var
   py     : TPythonObject;
begin
  py := InvokeMethod('msort',[]);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNDarrayGeneral.sort_complex():TNDarray ;
var
   py     : TPythonObject;
begin
  py := InvokeMethod('sort_complex',[]);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNDarrayGeneral.partition(kth: TArray<Integer>; axis : Integer = -1; kind : string = 'introselect'; order  : string  = null):TNDarray ;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple([TValue.FromArray<Integer>(kth)]);
  kwargs := TPyDict.Create;
  if (axis<>-1) then kwargs['axis'] := TNumPy.ToPython(axis);
  if (kind<>'introselect') then kwargs['kind'] := TNumPy.ToPython(kind);
  if (order<>null) then kwargs['order'] := TNumPy.ToPython(order);
  py := InvokeMethod('partition', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNDarrayGeneral.argpartition(kth: TArray<Integer>; axis: Integer = -1; kind : string = 'introselect'; order  : string  = null):TNDarray ;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple([TValue.FromArray<Integer>(kth)]);
  kwargs := TPyDict.Create;
  if (axis<>-1) then kwargs['axis'] := TNumPy.ToPython(axis);
  if (kind<>'introselect') then kwargs['kind'] := TNumPy.ToPython(kind);
  if (order<>null) then kwargs['order'] := TNumPy.ToPython(order);
  py := InvokeMethod('argpartition', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNDarrayGeneral.argmax(axis  : PInteger; var _out  :  TNDarray ):TNDarray ;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple();
  kwargs := TPyDict.Create;
  if (axis<>nil) then kwargs['axis'] := TNumPy.ToPython(axis^);
  if (_out <> nil) then kwargs['out'] := TNumPy.ToPython(_out);
  py := InvokeMethod('argmax', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNDarrayGeneral.nanargmax(axis  : PInteger = nil):TNDarray ;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple();
  kwargs := TPyDict.Create;
  if (axis<>nil) then kwargs['axis'] := TNumPy.ToPython(axis^);
  py := InvokeMethod('nanargmax', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNDarrayGeneral.argmin(axis  : PInteger ; var _out  :  TNDarray ):TNDarray ;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple();
  kwargs := TPyDict.Create;
  if (axis<>nil) then kwargs['axis'] := TNumPy.ToPython(axis^);
  if (_out <> nil) then kwargs['out'] := TNumPy.ToPython(_out);
  py := InvokeMethod('argmin', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNDarrayGeneral.nanargmin(axis  : PInteger = nil):TNDarray ;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple();
  kwargs := TPyDict.Create;
  if (axis<>nil) then kwargs['axis'] := TNumPy.ToPython(axis^);
  py := InvokeMethod('nanargmin', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNDarrayGeneral.argwhere:TNDarray ;
var
   py     : TPythonObject;
begin
  py := InvokeMethod('argwhere',[]);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNDarrayGeneral.flatnonzero:TNDarray ;
var
   py     : TPythonObject;
begin
  py := InvokeMethod('flatnonzero',[]);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function  TNDarrayGeneral.searchsorted(v :  TNDarray ; side: string = 'left'; sorter  :  TNDarray  = nil): TNDArray<Integer>;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple([v]);
  kwargs := TPyDict.Create;
  if (side<>'left') then kwargs['side'] := TNumPy.ToPython(side);
  if (sorter<>nil) then kwargs['sorter'] := TNumPy.ToPython(sorter);
  py := InvokeMethod('searchsorted', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDArray<Integer>>(py);
end;

Function TNDarrayGeneral.extract(arr :  TNDarray  ):TNDarray ;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple([arr]);
  kwargs := TPyDict.Create;
  py := InvokeMethod('extract', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function  TNDarrayGeneral.count_nonzero(axis: TArray<Integer>): TNDArray<Integer>;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple();
  kwargs := TPyDict.Create;
  if (Length(axis) > 0) then kwargs['axis'] := TNumPy.ToPython(TValue.FromArray<Integer>(axis));
  py := InvokeMethod('count_nonzero', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDArray<Integer>>(py);
end;

Function  TNDarrayGeneral.count_nonzero: Integer;
var
   py     : TPythonObject;
begin
  py := InvokeMethod('count_nonzero',[]);
  Result := TNumPy.ToCsharp<Integer>(py);
end;

Function TNDarrayGeneral.nanmin(axis: TArray<Integer> ; var _out  :  TNDarray  ; keepdims  : PBoolean  = nil):TNDarray ;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple();
  kwargs := TPyDict.Create;
  if (Length(axis) > 0) then kwargs['axis'] := TNumPy.ToPython(TValue.FromArray<Integer>(axis));
  if (_out <> nil) then kwargs['out'] := TNumPy.ToPython(_out);
  if (keepdims<>nil) then kwargs['keepdims'] := TNumPy.ToPython(keepdims^);
  py := InvokeMethod('nanmin', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNDarrayGeneral.nanmax(axis : TArray<Integer> ; var _out  :  TNDarray ; keepdims  : PBoolean  = nil):TNDarray ;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple();
  kwargs := TPyDict.Create;
  if (Length(axis) > 0) then kwargs['axis'] := TNumPy.ToPython(TValue.FromArray<Integer>(axis));
  if (_out <> nil) then kwargs['out'] := TNumPy.ToPython(_out);
  if (keepdims<>nil) then kwargs['keepdims'] := TNumPy.ToPython(keepdims^);
  py := InvokeMethod('nanmax', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNDarrayGeneral.ptp(axis: TArray<Integer> ; var _out  :  TNDarray  ; keepdims  : PBoolean  = nil):TNDarray ;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple();
  kwargs := TPyDict.Create;
  if (Length(axis) > 0) then kwargs['axis'] := TNumPy.ToPython(TValue.FromArray<Integer>(axis));
  if (_out <> nil) then kwargs['out'] := TNumPy.ToPython(_out);
  if (keepdims<>nil) then kwargs['keepdims'] := TNumPy.ToPython(keepdims^);
  py := InvokeMethod('ptp', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function  TNDarrayGeneral.percentile(q: TNDArray<Double>; axis: TArray<Integer>; var _out  :  TNDarray ; overwrite_input : Boolean = false;  interpolation : string= 'linear'; keepdims : Boolean = false):TNDArray<double>;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple([q]);
  kwargs := TPyDict.Create;
  if (Length(axis) > 0) then kwargs['axis'] := TNumPy.ToPython(TValue.FromArray<Integer>(axis));
  if (_out <> nil) then kwargs['out'] := TNumPy.ToPython(_out);
  if (overwrite_input<>false) then kwargs['overwrite_input'] := TNumPy.ToPython(overwrite_input);
  if (interpolation<>'linear') then kwargs['interpolation'] := TNumPy.ToPython(interpolation);
  if (keepdims<>false) then kwargs['keepdims'] := TNumPy.ToPython(keepdims);
  py := InvokeMethod('percentile', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDArray<double>>(py);
end;

Function  TNDarrayGeneral.percentile(q: TNDArray<Double>; var _out  :  TNDarray ; overwrite_input : Boolean = false;  interpolation : string = 'linear'): double;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple([q]);
  kwargs := TPyDict.Create;
  if (_out <> nil) then kwargs['out'] := TNumPy.ToPython(_out);
  if (overwrite_input<>false) then kwargs['overwrite_input'] := TNumPy.ToPython(overwrite_input);
  if (interpolation<>'linear') then kwargs['interpolation'] := TNumPy.ToPython(interpolation);
  py := InvokeMethod('percentile', pyargs, kwargs);
  Result := TNumPy.ToCsharp<double>(py);
end;

Function  TNDarrayGeneral.nanpercentile(q: TNDArray<Double>; axis: TArray<Integer>; var _out  :  TNDarray ; overwrite_input : Boolean = false;  interpolation : string = 'linear'; keepdims  : PBoolean  = nil): TNDArray<double>;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple([q]);
  kwargs := TPyDict.Create;
  if (Length(axis) >0) then kwargs['axis'] := TNumPy.ToPython(TValue.FromArray<Integer>(axis));
  if (_out <> nil) then kwargs['out'] := TNumPy.ToPython(_out);
  if (overwrite_input<>false) then kwargs['overwrite_input'] := TNumPy.ToPython(overwrite_input);
  if (interpolation<>'linear') then kwargs['interpolation'] := TNumPy.ToPython(interpolation);
  if (keepdims<>nil) then kwargs['keepdims'] := TNumPy.ToPython(keepdims^);
  py := InvokeMethod('nanpercentile', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDArray<double>>(py);
end;

Function  TNDarrayGeneral.nanpercentile(q: TNDArray<Double>; var _out  :  TNDarray  ; overwrite_input : Boolean = false; interpolation : string = 'linear') : Double;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple([q]);
  kwargs := TPyDict.Create;
  if (_out <> nil) then kwargs['out'] := TNumPy.ToPython(_out);
  if (overwrite_input<>false) then kwargs['overwrite_input'] := TNumPy.ToPython(overwrite_input);
  if (interpolation<>'linear') then kwargs['interpolation'] := TNumPy.ToPython(interpolation);
  py := InvokeMethod('nanpercentile', pyargs, kwargs);
  Result := TNumPy.ToCsharp<double>(py);
end;

Function  TNDarrayGeneral.quantile(q: TNDArray<Double>; axis : TArray<Integer>; var _out  :  TNDarray ; overwrite_input : Boolean = false; interpolation : string = 'linear'; keepdims: Boolean = false) : TNDArray<double>;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple([q]);
  kwargs := TPyDict.Create;
  if (Length(axis) > 0) then kwargs['axis'] := TNumPy.ToPython(TValue.FromArray<Integer>(axis));
  if (_out <> nil) then kwargs['out'] := TNumPy.ToPython(_out);
  if (overwrite_input<>false) then kwargs['overwrite_input'] := TNumPy.ToPython(overwrite_input);
  if (interpolation<>'linear') then kwargs['interpolation'] := TNumPy.ToPython(interpolation);
  if (keepdims<>false) then kwargs['keepdims'] := TNumPy.ToPython(keepdims);
  py := InvokeMethod('quantile', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDArray<double>>(py);
end;

Function  TNDarrayGeneral.quantile(q: TNDArray<Double>; var _out  :  TNDarray  ; overwrite_input : Boolean = false;  interpolation : string = 'linear') : Double;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple([q]);
  kwargs := TPyDict.Create;
  if (_out <> nil) then kwargs['out'] := TNumPy.ToPython(_out);
  if (overwrite_input<>false) then kwargs['overwrite_input'] := TNumPy.ToPython(overwrite_input);
  if (interpolation<>'linear') then kwargs['interpolation'] := TNumPy.ToPython(interpolation);
  py := InvokeMethod('quantile', pyargs, kwargs);
  Result := TNumPy.ToCsharp<double>(py);
end;

Function  TNDarrayGeneral.nanquantile(q: TNDArray<Double>; axis: TArray<Integer>; var _out  :  TNDarray ; overwrite_input: Boolean = false; interpolation : string = 'linear'; keepdims  : PBoolean  = nil): TNDArray<Double>;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple([q]);
  kwargs := TPyDict.Create;
  if (Length(axis) > 0) then kwargs['axis'] := TNumPy.ToPython(TValue.FromArray<Integer>(axis));
  if (_out <> nil) then kwargs['out'] := TNumPy.ToPython(_out);
  if (overwrite_input<>false) then kwargs['overwrite_input'] := TNumPy.ToPython(overwrite_input);
  if (interpolation<>'linear') then kwargs['interpolation'] := TNumPy.ToPython(interpolation);
  if (keepdims<>nil) then kwargs['keepdims'] := TNumPy.ToPython(keepdims^);
  py := InvokeMethod('nanquantile', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDArray<Double>>(py);
end;

Function  TNDarrayGeneral.nanquantile(q: TNDArray<Double>; var _out  :  TNDarray  ; overwrite_input : Boolean = false; interpolation : string = 'linear') : Double;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple([q]);
  kwargs := TPyDict.Create;
  if (_out <> nil) then kwargs['out'] := TNumPy.ToPython(_out);
  if (overwrite_input<>false) then kwargs['overwrite_input'] := TNumPy.ToPython(overwrite_input);
  if (interpolation<>'linear') then kwargs['interpolation'] := TNumPy.ToPython(interpolation);
  py := InvokeMethod('nanquantile', pyargs, kwargs);
  Result := TNumPy.ToCsharp<double>(py);
end;

Function  TNDarrayGeneral.median(axis: TArray<Integer>; var _out  :  TNDarray  ; overwrite_input : Boolean = false; keepdims : Boolean = false): TNDArray<Double>;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple();
  kwargs := TPyDict.Create;
  if (Length(axis) > 0) then kwargs['axis'] := TNumPy.ToPython(TValue.FromArray<Integer>(axis));
  if (_out <> nil) then kwargs['out'] := TNumPy.ToPython(_out);
  if (overwrite_input<>false) then kwargs['overwrite_input'] := TNumPy.ToPython(overwrite_input);
  if (keepdims<>false) then kwargs['keepdims'] := TNumPy.ToPython(keepdims);
  py := InvokeMethod('median', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDArray<Double>>(py);
end;

Function  TNDarrayGeneral.median(var _out  :  TNDarray ; overwrite_input : Boolean = false): double;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple();
  kwargs := TPyDict.Create;
  if (_out <> nil) then kwargs['out'] := TNumPy.ToPython(_out);
  if (overwrite_input<>false) then kwargs['overwrite_input'] := TNumPy.ToPython(overwrite_input);
  py := InvokeMethod('median', pyargs, kwargs);
  Result := TNumPy.ToCsharp<double>(py);
end;

Function  TNDarrayGeneral.average(axis: TArray<Integer>; weights  :  TNDarray  = nil; returned : Boolean = false): TNDArray<double>;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple();
  kwargs := TPyDict.Create;
  if (Length(axis) > 0) then kwargs['axis'] := TNumPy.ToPython(TValue.FromArray<Integer>(axis));
  if (weights<>nil) then kwargs['weights'] := TNumPy.ToPython(weights);
  if (returned<>false) then kwargs['returned'] := TNumPy.ToPython(returned);
  py := InvokeMethod('average', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDArray<double>>(py);
end;

Function  TNDarrayGeneral.average(weights  :  TNDarray  = nil; returned : Boolean = false): double;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple();
  kwargs := TPyDict.Create;
  if (weights<>nil) then kwargs['weights'] := TNumPy.ToPython(weights);
  if (returned<>false) then kwargs['returned'] := TNumPy.ToPython(returned);
  py := InvokeMethod('average', pyargs, kwargs);
  Result := TNumPy.ToCsharp<double>(py);
end;

Function  TNDarrayGeneral.mean(axis: TArray<Integer>; dtype  : TDtype ; var _out  :  TNDarray ; keepdims  : PBoolean  = nil): TNDArray<double>;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple();
  kwargs := TPyDict.Create;
  if (Length(axis) > 0) then kwargs['axis'] := TNumPy.ToPython(TValue.FromArray<integer>(axis));
  if (dtype<>nil) then kwargs['dtype'] := TNumPy.ToPython(dtype);
  if (_out <> nil) then kwargs['out'] := TNumPy.ToPython(_out);
  if (keepdims<>nil) then kwargs['keepdims'] := TNumPy.ToPython(keepdims^);
  py := InvokeMethod('mean', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDArray<double>>(py);
end;

Function  TNDarrayGeneral.mean(dtype  : TDtype  ; var _out  :  TNDarray  ): double;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple();
  kwargs := TPyDict.Create;
  if (dtype<>nil) then kwargs['dtype'] := TNumPy.ToPython(dtype);
  if (_out <> nil) then kwargs['out'] := TNumPy.ToPython(_out);
  py := InvokeMethod('mean', pyargs, kwargs);
  Result := TNumPy.ToCsharp<double>(py);
end;

Function  TNDarrayGeneral.std(axis:TArray<Integer>; dtype  : TDtype  ; var _out  :  TNDarray ; ddof: Integer = 0; keepdims  : PBoolean  = nil):TNDArray<double>;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple();
  kwargs := TPyDict.Create;
  if (Length(axis) > 0) then kwargs['axis'] := TNumPy.ToPython(TValue.FromArray<Integer>(axis));
  if (dtype<>nil) then kwargs['dtype'] := TNumPy.ToPython(dtype);
  if (_out <> nil) then kwargs['out'] := TNumPy.ToPython(_out);
  if (ddof<>0) then kwargs['ddof'] := TNumPy.ToPython(ddof);
  if (keepdims<>nil) then kwargs['keepdims'] := TNumPy.ToPython(keepdims^);
  py := InvokeMethod('std', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDArray<double>>(py);
end;

Function  TNDarrayGeneral.std(dtype  : TDtype ; var _out  :  TNDarray  ; ddof : Integer = 0): Double;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple();
  kwargs := TPyDict.Create;
  if (dtype<>nil) then kwargs['dtype'] := TNumPy.ToPython(dtype);
  if (_out <> nil) then kwargs['out'] := TNumPy.ToPython(_out);
  if (ddof<>0) then kwargs['ddof'] := TNumPy.ToPython(ddof);
  py := InvokeMethod('std', pyargs, kwargs);
  Result := TNumPy.ToCsharp<double>(py);
end;

Function  TNDarrayGeneral.&var(axis: TArray<Integer>; dtype  : TDtype  ; var _out  :  TNDarray  ; ddof : Integer = 0; keepdims  : PBoolean  = nil):TNDArray<double>;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple();
  kwargs := TPyDict.Create;
  if (Length(axis) > 0) then kwargs['axis'] := TNumPy.ToPython(TValue.FromArray<Integer>(axis));
  if (dtype<>nil) then kwargs['dtype'] := TNumPy.ToPython(dtype);
  if (_out <> nil) then kwargs['out'] := TNumPy.ToPython(_out);
  if (ddof<>0) then kwargs['ddof'] := TNumPy.ToPython(ddof);
  if (keepdims<>nil) then kwargs['keepdims'] := TNumPy.ToPython(keepdims^);
  py := InvokeMethod('var', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDArray<double>>(py);
end;

Function  TNDarrayGeneral.&var(dtype  : TDtype  ; var _out  :  TNDarray  ; ddof : Integer = 0): double;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple();
  kwargs := TPyDict.Create;
  if (dtype<>nil) then kwargs['dtype'] := TNumPy.ToPython(dtype);
  if (_out <> nil) then kwargs['out'] := TNumPy.ToPython(_out);
  if (ddof<>0) then kwargs['ddof'] := TNumPy.ToPython(ddof);
  py := InvokeMethod('var', pyargs, kwargs);
  Result := TNumPy.ToCsharp<double>(py);
end;

Function  TNDarrayGeneral.nanmedian(axis: TArray<Integer>; var _out  :  TNDarray ; overwrite_input : Boolean = false; keepdims  : PBoolean  = nil): TNDArray<double>;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple();
  kwargs := TPyDict.Create;
  if (Length(axis) > 0) then kwargs['axis'] := TNumPy.ToPython(TValue.FromArray<Integer>(axis));
  if (_out <> nil) then kwargs['out'] := TNumPy.ToPython(_out);
  if (overwrite_input<>false) then kwargs['overwrite_input'] := TNumPy.ToPython(overwrite_input);
  if (keepdims<>nil) then kwargs['keepdims'] := TNumPy.ToPython(keepdims^);
  py := InvokeMethod('nanmedian', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDArray<double>>(py);
end;

Function  TNDarrayGeneral.nanmedian(var _out  :  TNDarray  ; overwrite_input : Boolean = false): double;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple();
  kwargs := TPyDict.Create;
  if (_out <> nil) then kwargs['out'] := TNumPy.ToPython(_out);
  if (overwrite_input<>false) then kwargs['overwrite_input'] := TNumPy.ToPython(overwrite_input);
  py := InvokeMethod('nanmedian', pyargs, kwargs);
  Result := TNumPy.ToCsharp<double>(py);
end;

Function  TNDarrayGeneral.nanmean(axis: TArray<Integer>; dtype  : TDtype ; var _out  :  TNDarray  ; keepdims  : PBoolean  = nil):TNDArray<double>;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple();
  kwargs := TPyDict.Create;
  if (Length(axis) > 0) then kwargs['axis'] := TNumPy.ToPython(TValue.FromArray<Integer>(axis));
  if (dtype<>nil) then kwargs['dtype'] := TNumPy.ToPython(dtype);
  if (_out <> nil) then kwargs['out'] := TNumPy.ToPython(_out);
  if (keepdims<>nil) then kwargs['keepdims'] := TNumPy.ToPython(keepdims^);
  py := InvokeMethod('nanmean', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray<double>>(py);
end;

Function  TNDarrayGeneral.nanmean(dtype  : TDtype  ; var _out  :  TNDarray ): double;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple();
  kwargs := TPyDict.Create;
  if (dtype<>nil) then kwargs['dtype'] := TNumPy.ToPython(dtype);
  if (_out <> nil) then kwargs['out'] := TNumPy.ToPython(_out);
  py := InvokeMethod('nanmean', pyargs, kwargs);
  Result := TNumPy.ToCsharp<double>(py);
end;

Function  TNDarrayGeneral.nanstd(axis: TArray<Integer>; dtype  : TDtype ; var _out  :  TNDarray ; ddof : Integer = 0; keepdims  : PBoolean  = nil): TNDarray<double>;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple();
  kwargs := TPyDict.Create;
  if (Length(axis) > 0) then kwargs['axis'] := TNumPy.ToPython(TValue.FromArray<Integer>(axis));
  if (dtype<>nil) then kwargs['dtype'] := TNumPy.ToPython(dtype);
  if (_out <> nil) then kwargs['out'] := TNumPy.ToPython(_out);
  if (ddof<>0) then kwargs['ddof'] := TNumPy.ToPython(ddof);
  if (keepdims<>nil) then kwargs['keepdims'] := TNumPy.ToPython(keepdims^);
  py := InvokeMethod('nanstd', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray<double>>(py);
end;

Function  TNDarrayGeneral.nanstd(dtype  : TDtype  ; var _out  :  TNDarray  ; ddof: Integer  = 0): double;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple();
  kwargs := TPyDict.Create;
  if (dtype<>nil) then kwargs['dtype'] := TNumPy.ToPython(dtype);
  if (_out <> nil) then kwargs['out'] := TNumPy.ToPython(_out);
  if (ddof<>0) then kwargs['ddof'] := TNumPy.ToPython(ddof);
  py := InvokeMethod('nanstd', pyargs, kwargs);
  Result := TNumPy.ToCsharp<double>(py);
end;

Function  TNDarrayGeneral.nanvar(axis: TArray<Integer>; dtype  : TDtype  ; var _out  :  TNDarray ; ddof: Integer = 0; keepdims  : PBoolean  = nil): TNDarray<double>;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple();
  kwargs := TPyDict.Create;
  if (Length(axis) > 0) then kwargs['axis'] := TNumPy.ToPython(TValue.FromArray<Integer>(axis));
  if (dtype<>nil) then kwargs['dtype'] := TNumPy.ToPython(dtype);
  if (_out <> nil) then kwargs['out'] := TNumPy.ToPython(_out);
  if (ddof<>0) then kwargs['ddof'] := TNumPy.ToPython(ddof);
  if (keepdims<>nil) then kwargs['keepdims'] := TNumPy.ToPython(keepdims^);
  py := InvokeMethod('nanvar', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray<double>>(py);
end;

Function  TNDarrayGeneral.nanvar(dtype  : TDtype  ; var _out  :  TNDarray  ; ddof: Integer = 0): Double;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple();
  kwargs := TPyDict.Create;
  if (dtype<>nil) then kwargs['dtype'] := TNumPy.ToPython(dtype);
  if (_out <> nil) then kwargs['out'] := TNumPy.ToPython(_out);
  if (ddof<>0) then kwargs['ddof'] := TNumPy.ToPython(ddof);
  py := InvokeMethod('nanvar', pyargs, kwargs);
  Result := TNumPy.ToCsharp<double>(py);
end;

Function TNDarrayGeneral.corrcoef(y  :  TNDarray  = nil; rowvar: Boolean = true):TNDarray ;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple();
  kwargs := TPyDict.Create;
  if (y<>nil) then kwargs['y'] := TNumPy.ToPython(y);
  if (rowvar<>true) then kwargs['rowvar'] := TNumPy.ToPython(rowvar);
  py := InvokeMethod('corrcoef', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNDarrayGeneral.correlate(a :  TNDarray ; mode: string = 'valid'):TNDarray ;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple([a]);
  kwargs := TPyDict.Create;
  if (mode<>'valid') then kwargs['mode'] := TNumPy.ToPython(mode);
  py := InvokeMethod('correlate', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function TNDarrayGeneral.cov(y  :  TNDarray  = nil; rowvar : Boolean = true; bias : Boolean = false; ddof  : PInteger = nil; fweights  :  TNDarray  = nil; aweights  :  TNDarray  = nil):TNDarray ;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple();
  kwargs := TPyDict.Create;
  if (y<>nil) then kwargs['y'] := TNumPy.ToPython(y);
  if (rowvar<>true) then kwargs['rowvar'] := TNumPy.ToPython(rowvar);
  if (bias<>false) then kwargs['bias'] := TNumPy.ToPython(bias);
  if (ddof<>nil) then kwargs['ddof'] := TNumPy.ToPython(ddof^);
  if (fweights<>nil) then kwargs['fweights'] := TNumPy.ToPython(fweights);
  if (aweights<>nil) then kwargs['aweights'] := TNumPy.ToPython(aweights);
  py := InvokeMethod('cov', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;


Function TNDarrayGeneral.histogram(bins  : PInteger = nil; range: TArray<Double> = []; normed  : PBoolean  = nil; weights  :  TNDarray  = nil; density  : PBoolean  = nil):TArray<TNDarray>;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
   res    : TArray<TPythonObject>;
begin
  pyargs := TNumPy.ToTuple();
  kwargs := TPyDict.Create;
  if (bins<>nil) then kwargs['bins'] := TNumPy.ToPython(bins^);
  if (Length(range) > 0) then kwargs['range'] := TNumPy.ToPython(TValue.FromArray<Double>(range));
  if (normed<>nil) then kwargs['normed'] := TNumPy.ToPython(normed^);
  if (weights<>nil) then kwargs['weights'] := TNumPy.ToPython(weights);
  if (density<>nil) then kwargs['density'] := TNumPy.ToPython(density^);
  py := InvokeMethod('histogram', pyargs, kwargs);

  res := py.AsArrayofPyObj;

  SetLength(Result,2);
  Result[0] := TNumPy.ToCsharp<TNDarray>(res[0]);
  Result[1] := TNumPy.ToCsharp<TNDarray>(res[1]);

end;


Function TNDarrayGeneral.histogram(bins  :  TNDarray  = nil; range: TArray<Double> = []; normed  : PBoolean  = nil; weights  :  TNDarray  = nil; density  : PBoolean  = nil):TArray<TNDarray>;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
   res    : TArray<TPythonObject>;
begin
  pyargs := TNumPy.ToTuple();
  kwargs := TPyDict.Create;
  if (bins<>nil) then kwargs['bins'] := TNumPy.ToPython(bins);
  if (Length(range) > 0) then kwargs['range'] := TNumPy.ToPython(TValue.FromArray<Double>(range));
  if (normed<>nil) then kwargs['normed'] := TNumPy.ToPython(normed^);
  if (weights<>nil) then kwargs['weights'] := TNumPy.ToPython(weights);
  if (density<>nil) then kwargs['density'] := TNumPy.ToPython(density^);
  py := InvokeMethod('histogram', pyargs, kwargs);

  res := py.AsArrayofPyObj;

  SetLength(Result,2);
  Result[0] := TNumPy.ToCsharp<TNDarray>(res[0]);
  Result[1] := TNumPy.ToCsharp<TNDarray>(res[1]);
end;


Function TNDarrayGeneral.histogram(bins : TList<string> = nil; range: TArray<Double> = []; normed  : PBoolean  = nil; weights  :  TNDarray  = nil; density  : PBoolean  = nil):TArray<TNDarray>;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
   res    : TArray<TPythonObject>;
begin
  pyargs := TNumPy.ToTuple();
  kwargs := TPyDict.Create;
  if (bins<>nil) then kwargs['bins'] := TNumPy.ToPython(bins);
  if (Length(range) > 0) then kwargs['range'] := TNumPy.ToPython(TValue.FromArray<Double>(range));
  if (normed<>nil) then kwargs['normed'] := TNumPy.ToPython(normed^);
  if (weights<>nil) then kwargs['weights'] := TNumPy.ToPython(weights);
  if (density<>nil) then kwargs['density'] := TNumPy.ToPython(density^);
  py := InvokeMethod('histogram', pyargs, kwargs);

  res := py.AsArrayofPyObj;

  SetLength(Result,2);
  Result[0] := TNumPy.ToCsharp<TNDarray>(res[0]);
  Result[1] := TNumPy.ToCsharp<TNDarray>(res[1]);
end;


Function TNDarrayGeneral.histogram2d(y :  TNDarray ; bins  : PInteger = nil; range: TArray<Double> = []; density  : PBoolean  = nil; normed  : PBoolean  = nil; weights  :  TNDarray  = nil):TArray<TNDarray>;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
   res    : TArray<TPythonObject>;
begin
  pyargs := TNumPy.ToTuple([y]);
  kwargs := TPyDict.Create;
  if (bins<>nil) then kwargs['bins'] := TNumPy.ToPython(bins^);
  if (Length(range) > 0) then kwargs['range'] := TNumPy.ToPython(TValue.FromArray<Double>(range));
  if (density<>nil) then kwargs['density'] := TNumPy.ToPython(density^);
  if (normed<>nil) then kwargs['normed'] := TNumPy.ToPython(normed^);
  if (weights<>nil) then kwargs['weights'] := TNumPy.ToPython(weights);
  py := InvokeMethod('histogram2d', pyargs, kwargs);

  res := py.AsArrayofPyObj;

  SetLength(Result,3);
  Result[0] := TNumPy.ToCsharp<TNDarray>(res[0]);
  Result[1] := TNumPy.ToCsharp<TNDarray>(res[1]);
  Result[2] := TNumPy.ToCsharp<TNDarray>(res[2]);
end;


Function TNDarrayGeneral.histogram2d(y :  TNDarray ; bins  :  TNDarray  = nil; range: TArray<Double> = []; density  : PBoolean  = nil; normed  : PBoolean  = nil; weights  :  TNDarray  = nil):TArray<TNDarray>;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
   res    : TArray<TPythonObject>;
begin
  pyargs := TNumPy.ToTuple([y]);
  kwargs := TPyDict.Create;
  if (bins<>nil) then kwargs['bins'] := TNumPy.ToPython(bins);
  if (Length(range) > 0) then kwargs['range'] := TNumPy.ToPython(TValue.FromArray<Double>(range));
  if (density<>nil) then kwargs['density'] := TNumPy.ToPython(density^);
  if (normed<>nil) then kwargs['normed'] := TNumPy.ToPython(normed^);
  if (weights<>nil) then kwargs['weights'] := TNumPy.ToPython(weights);
  py := InvokeMethod('histogram2d', pyargs, kwargs);

  res := py.AsArrayofPyObj;

  SetLength(Result,3);
  Result[0] := TNumPy.ToCsharp<TNDarray>(res[0]);
  Result[1] := TNumPy.ToCsharp<TNDarray>(res[1]);
  Result[2] := TNumPy.ToCsharp<TNDarray>(res[2]);
end;


Function TNDarrayGeneral.histogram2d(y :  TNDarray ; bins : TList<string> = nil; range: TArray<Double> = []; density  : PBoolean  = nil; normed  : PBoolean  = nil; weights  :  TNDarray  = nil):TArray<TNDarray>;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
   res    : TArray<TPythonObject>;
begin
  pyargs := TNumPy.ToTuple([y]);
  kwargs := TPyDict.Create;
  if (bins<>nil) then kwargs['bins'] := TNumPy.ToPython(bins);
  if (Length(range) > 0) then kwargs['range'] := TNumPy.ToPython(TValue.FromArray<Double>(range));
  if (density<>nil) then kwargs['density'] := TNumPy.ToPython(density^);
  if (normed<>nil) then kwargs['normed'] := TNumPy.ToPython(normed^);
  if (weights<>nil) then kwargs['weights'] := TNumPy.ToPython(weights);
  py := InvokeMethod('histogram2d', pyargs, kwargs);

  res := py.AsArrayofPyObj;

  SetLength(Result,3);
  Result[0] := TNumPy.ToCsharp<TNDarray>(res[0]);
  Result[1] := TNumPy.ToCsharp<TNDarray>(res[1]);
  Result[2] := TNumPy.ToCsharp<TNDarray>(res[2]);
end;


Function TNDarrayGeneral.histogramdd(bins  : PInteger = nil; range: TArray<Double> = []; density  : PBoolean  = nil; normed  : PBoolean  = nil; weights  :  TNDarray  = nil):TArray<TNDarray>;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
   res    : TArray<TPythonObject>;
begin
  pyargs := TNumPy.ToTuple();
  kwargs := TPyDict.Create;
  if (bins<>nil) then kwargs['bins'] := TNumPy.ToPython(bins^);
  if (Length(range) > 0) then kwargs['range'] := TNumPy.ToPython(TValue.FromArray<Double>(range));
  if (density<>nil) then kwargs['density'] := TNumPy.ToPython(density^);
  if (normed<>nil) then kwargs['normed'] := TNumPy.ToPython(normed^);
  if (weights<>nil) then kwargs['weights'] := TNumPy.ToPython(weights);
  py := InvokeMethod('histogramdd', pyargs, kwargs);

  res := py.AsArrayofPyObj;

  SetLength(Result,2);
  Result[0] := TNumPy.ToCsharp<TNDarray>(res[0]);
  Result[1] := TNumPy.ToCsharp<TNDarray>(res[1]);

end;


Function TNDarrayGeneral.histogramdd(bins  :  TNDarray  = nil; range: TArray<Double> = []; density  : PBoolean  = nil; normed  : PBoolean  = nil; weights  :  TNDarray  = nil):TArray<TNDarray>;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
   res    : TArray<TPythonObject>;
begin
  pyargs := TNumPy.ToTuple();
  kwargs := TPyDict.Create;
  if (bins<>nil) then kwargs['bins'] := TNumPy.ToPython(bins);
  if  (Length(range) > 0) then kwargs['range'] := TNumPy.ToPython(TValue.FromArray<Double>(range));
  if (density<>nil) then kwargs['density'] := TNumPy.ToPython(density^);
  if (normed<>nil) then kwargs['normed'] := TNumPy.ToPython(normed^);
  if (weights<>nil) then kwargs['weights'] := TNumPy.ToPython(weights);
  py := InvokeMethod('histogramdd', pyargs, kwargs);

  res := py.AsArrayofPyObj;

  SetLength(Result,2);
  Result[0] := TNumPy.ToCsharp<TNDarray>(res[0]);
  Result[1] := TNumPy.ToCsharp<TNDarray>(res[1]);
end;


Function TNDarrayGeneral.histogramdd(bins : TList<string> = nil; range: TArray<Double> = []; density  : PBoolean  = nil; normed  : PBoolean  = nil; weights  :  TNDarray  = nil):TArray<TNDarray>;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
   res    : TArray<TPythonObject>;
begin
  pyargs := TNumPy.ToTuple();
  kwargs := TPyDict.Create;
  if (bins<>nil) then kwargs['bins'] := TNumPy.ToPython(bins);
  if(Length(range) > 0) then kwargs['range'] := TNumPy.ToPython(TValue.FromArray<Double>(range));
  if (density<>nil) then kwargs['density'] := TNumPy.ToPython(density^);
  if (normed<>nil) then kwargs['normed'] := TNumPy.ToPython(normed^);
  if (weights<>nil) then kwargs['weights'] := TNumPy.ToPython(weights);
  py := InvokeMethod('histogramdd', pyargs, kwargs);

  res := py.AsArrayofPyObj;

  SetLength(Result,2);
  Result[0] := TNumPy.ToCsharp<TNDarray>(res[0]);
  Result[1] := TNumPy.ToCsharp<TNDarray>(res[1]);
end;

Function TNDarrayGeneral.bincount(weights  :  TNDarray  = nil; minlength : Integer = 0):TNDarray ;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple();
  kwargs := TPyDict.Create;
  if (weights<>nil) then kwargs['weights'] := TNumPy.ToPython(weights);
  if (minlength<>0) then kwargs['minlength'] := TNumPy.ToPython(minlength);
  py := InvokeMethod('bincount', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;

Function  TNDarrayGeneral.histogram_bin_edges(bins  : PInteger = nil; range: TArray<Double> = []; weights  :  TNDarray  = nil): TNDarray<Double>;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple();
  kwargs := TPyDict.Create;
  if (bins<>nil) then kwargs['bins'] := TNumPy.ToPython(bins^);
  if (Length(range) > 0) then kwargs['range'] := TNumPy.ToPython(TValue.FromArray<Double>(range));
  if (weights<>nil) then kwargs['weights'] := TNumPy.ToPython(weights);
  py := InvokeMethod('histogram_bin_edges', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray<Double>>(py);
end;

Function  TNDarrayGeneral.histogram_bin_edges(bins  :  TNDarray  = nil; range: TArray<Double> = []; weights  :  TNDarray  = nil):TNDarray<Double>;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple();
  kwargs := TPyDict.Create;
  if (bins<>nil) then kwargs['bins'] := TNumPy.ToPython(bins);
  if(Length(range) > 0) then kwargs['range'] := TNumPy.ToPython(TValue.FromArray<Double>(range));
  if (weights<>nil) then kwargs['weights'] := TNumPy.ToPython(weights);
  py := InvokeMethod('histogram_bin_edges', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray<Double>>(py);
end;

Function  TNDarrayGeneral.histogram_bin_edges(bins: TList<string> = nil;  range: TArray<Double> = [];  weights  :  TNDarray  = nil):TNDarray<Double>;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple();
  kwargs := TPyDict.Create;
  if (bins<>nil) then kwargs['bins'] := TNumPy.ToPython(bins);
  if (Length(range) > 0) then kwargs['range'] := TNumPy.ToPython(TValue.FromArray<Double>(range));
  if (weights<>nil) then kwargs['weights'] := TNumPy.ToPython(weights);
  py := InvokeMethod('histogram_bin_edges', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray<Double>>(py);
end;

Function  TNDarrayGeneral.digitize(bins :  TNDarray ; right : Boolean = false):TNDarray ;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   py     : TPythonObject;
begin
  pyargs := TNumPy.ToTuple([bins]);
  kwargs := TPyDict.Create;
  if (right<>false) then kwargs['right'] := TNumPy.ToPython(right);
  py := InvokeMethod('digitize', pyargs, kwargs);
  Result := TNumPy.ToCsharp<TNDarray>(py);
end;


end.