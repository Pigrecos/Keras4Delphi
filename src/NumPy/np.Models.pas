{*******************************************************}
{                                                       }
{       Numpy Wrapper for Delphi                        }
{                                                       }
{       Copyright (C) 2020 Pigrecos(Max)                }
{                                                       }
{*******************************************************}
unit np.Models;



interface
  uses System.SysUtils, System.Rtti,System.Generics.Collections,
       PythonEngine,
       Python.Utils;

const
  NULL_ : Integer = $CC;

type

  TDtype = class(TPythonObject)
   public
     constructor Create(t: TDtype); overload;
     constructor Create(_pyobject: PPyObject);overload;
  end;

  TFlags = class(TPythonObject)
   public
     constructor Create(_pyobject: PPyObject);
  end;

  TMatrix = class(TPythonObject)
   public
     constructor Create(_pyobject: PPyObject);
  end;

  TMemMapMode = class(TPythonObject)
   public
     constructor Create(_pyobject: PPyObject);
  end;

  PTnp_Shape = ^Tnp_Shape;
  Tnp_Shape = record
  private
    function GetItem(index:Integer): Integer;
   public
     Dimensions : TArray<Integer>;

     constructor Create(shape : TArray<Integer>);
     function ToString: string;

     class operator Equal   (a,b: Tnp_Shape): Boolean;
     class operator NotEqual(a,b: Tnp_Shape): Boolean;

     property Item[index: Integer]: Integer read GetItem;default;
  end;

  /// <summary>
  /// NDArray can be indexed using slicing
  /// A slice is constructed by start:stop:step notation
  ///
  /// Examples:
  ///
  /// a[start:stop]  # items start through stop-1
  /// a[start:]      # items start through the rest of the array
  /// a[:stop]       # items from the beginning through stop-1
  ///
  /// The key point to remember is that the :stop value represents the first value that is not
  /// in the selected slice. So, the difference between stop and start is the number of elements
  /// selected (if step is 1, the default).
  ///
  /// There is also the step value, which can be used with any of the above:
  /// a[:]           # a copy of the whole array
  /// a[start:stop:step] # start through not past stop, by step
  ///
  /// The other feature is that start or stop may be a negative number, which means it counts
  /// from the end of the array instead of the beginning. So:
  /// a[-1]    # last item in the array
  /// a[-2:]   # last two items in the array
  /// a[:-2]   # everything except the last two items
  /// Similarly, step may be a negative number:
  ///
  /// a[::- 1]    # all items in the array, reversed
  /// a[1::- 1]   # the first two items, reversed
  /// a[:-3:-1]  # the last two items, reversed
  /// a[-3::- 1]  # everything except the last two items, reversed
  ///
  /// NumSharp is kind to the programmer if there are fewer items than
  /// you ask for. For example, if you  ask for a[:-2] and a only contains one element, you get an
  /// empty list instead of an error.Sometimes you would prefer the error, so you have to be aware
  /// that this may happen.
  ///
  /// Adapted from Greg Hewgill's answer on Stackoverflow: https://stackoverflow.com/questions/509211/understanding-slice-notation
  ///
  /// Note: special IsIndex == true
  /// It will pick only a single value at Start in this dimension effectively reducing the Shape of the sliced matrix by 1 dimension.
  /// It can be used to reduce an N-dimensional array/matrix to a (N-1)-dimensional array/matrix
  ///
  /// Example:
  /// a=[[1, 2], [3, 4]]
  /// a[:, 1] returns the second column of that 2x2 matrix as a 1-D vector
  /// </summary>
  Tnp_Slice = record
   public
     Start   : Integer;
     Stop    : Integer;
     Step    : Integer;
     IsIndex : Boolean;

     len     : Integer;

     function     Create(_Start: Integer =$CC; _Stop: Integer = $CC; _Step: Integer = 1 ):Tnp_Slice; overload;
     function     Create(slice_notation : string ):Tnp_Slice; overload;
     function     ParseSlices(multi_slice_notation : string ):TArray<Tnp_Slice>;
     function     FormatSlices(slices : TArray<Tnp_Slice>): string ;
     procedure    Parse(slice_notation : string );
     function     All: Tnp_Slice;
     function     Index(index: Integer): Tnp_Slice;
     function     ToString:string;
     function     GetSize(dim: Integer):Integer;
     function     GetAbsStep: Integer;
     function     GetAbsStart(dim: Integer): Integer;
     function     GetAbsStop(dim: Integer): Integer;
     function     ToPython: TPythonObject;
     function     FormatNullableIntForPython(i : Integer=0): string;

     class operator Equal   (a,b: Tnp_Slice): Boolean;
     class operator NotEqual(a,b: Tnp_Slice): Boolean;
  end;

  TNDArray<T>  = class;

  TNDArray = class(TPythonObject)
    private

      function  Getitem(index: string): TNDarray; overload;
      procedure SetItem(index: string; const Value: TNDarray); overload;

      function  Getitem(index: TArray<Integer>): TNDarray; overload;
      procedure SetItem(index: TArray<Integer>; const Value: TNDarray); overload;

      function  Getitem(index: TArray<TNDarray>): TNDarray; overload;
      procedure SetItem(index:TArray<TNDarray>; const Value: TNDarray); overload;

      function  Getitem(index: TArray<TVarRec>): TNDarray; overload;
      procedure SetItem(index: TArray<TVarRec>; const Value: TNDarray); overload;

    public
      constructor Create(t: TNDArray); overload;
      constructor Create(_pyobject: PPyObject);overload;

      function  flags : TFlags ; // TODO: implement Flags
      function  shape : Tnp_Shape ;
      function  strides : TArray<Integer>;
      function  ndim : Integer;
      function  data : PPyObject;
      function  size : Integer;
      function  itemsize : Integer;
      function  nbytes: Integer;
      function  base : TNDArray;
      function  dtype : TDtype;
      function  T : TNDArray;
      //public NDarray real => new NDarray(self.GetAttr("real"));
      //public NDarray imag => new NDarray(self.GetAttr("imag"));
      function  flat : PPyObject; // todo: wrap and support usecases
      function  ctypes : PPyObject; // TODO: wrap ctypes
      function  len : Integer;
      procedure itemset(args: TArray<TValue>);
      function  ToString(order: PChar): TArray<Byte>; overload;
      function  ToBytes(order : PChar = nil): TArray<Byte>;
      procedure view(dtype : TDtype ; tipo : Variant) ;
      procedure resize(new_shape : Tnp_Shape; refcheck : PBoolean = nil) ;
      function  reshape(newshape: TArray<Integer>): TNDarray ; overload;
      function  repr : string;
      function  str: string;
      function asscalar<T>: T;

      //------------------------------
      // Comparison operators:
      //------------------------------
      function  equal     (obj: TValue): TNDarray<Boolean> ;
      function  not_equals(obj: TValue): TNDarray<Boolean>;

      class function  opLess   (a: TNDarray; obj: TValue) : TNDArray<Boolean>;
      class function  opLessEq (a: TNDarray; obj: TValue) : TNDArray<Boolean>;
      class function  opGreat  (a: TNDarray; obj: TValue) : TNDArray<Boolean>;
      class function  opGreatEq(a: TNDarray; obj: TValue) : TNDArray<Boolean>;

      //------------------------------
      // Truth value of an array(bool) :
      //------------------------------
      class function nonzero(a: TNDarray): TNDarray<Boolean>;

      //------------------------------
      // Unary operations:
      //------------------------------
      class function opNeg(a: TNDarray): TNDarray;
      class function opPos(a: TNDarray): TNDarray;
      class function opNot(a: TNDarray): TNDarray;

      //------------------------------
      // Arithmetic operators:
      //------------------------------
      Class Function opAdd(a :  TNDarray ; obj: TValue)   : TNDarray ; overload;
      Class Function opAdd(obj: TValue;     a : TNDarray ): TNDarray ; overload;
		  Class Function opAdd(a :  TNDarray ; obj: TNDarray ): TNDarray ; overload;
      Class Function opSub(a :  TNDarray ; obj: TValue)   : TNDarray ; overload;
		  Class Function opSub(a :  TNDarray ; obj: TNDarray ): TNDarray ; overload;
      Class Function opMul(a :  TNDarray ; obj: TValue)   : TNDarray ; overload;
      Class Function opMul(obj: TValue;    a  : TNDarray ): TNDarray ; overload;
		  Class Function opMul(a :  TNDarray ; obj: TNDarray ): TNDarray ; overload;
      Class Function opDiv(a :  TNDarray ; obj: TValue)   : TNDarray ; overload;
      Class Function opDiv(a :  TNDarray ; obj: TNDarray ): TNDarray ; overload;

		  Function opDivMod (obj: TValue): TNDarray ;
      Function opPow    (obj: TValue): TNDarray ;

		  Class Function floordiv (a  : TNDarray ; obj: TValue) : TNDarray ;
      Class Function opMod    (a  : TNDarray ; obj: TValue) : TNDarray ;
      Class Function opSLeft  (a  : TNDarray ; obj: integer): TNDarray ;
      Class Function opSRight (a  : TNDarray ; obj: integer): TNDarray ;
      Class Function opAnd    (a  : TNDarray ; obj: integer): TNDarray ;
      Class Function opOr     (a  : TNDarray ; obj: integer): TNDarray ;
      Class Function opXor    (a  : TNDarray ; obj: integer): TNDarray ;

      //------------------------------
      // Arithmetic, in-place:
      //------------------------------
      Function opIAdd(obj: TValue)   : TNDarray ;
      Function opISub(obj: TValue)   : TNDarray ;
		  Function opIMul(obj: TValue)   : TNDarray ;
      Function opIDiv(obj: TValue)   : TNDarray ;

      Function opITrueDiv(obj: TValue ): TNDarray ;
      Function IFloorDiv (obj: TValue) : TNDarray ;

      Function opIMod(obj: TValue)   : TNDarray ;
      Function opIPow(obj: TValue)   : TNDarray ;

      Function IopSLeft  (obj: TValue): TNDarray ;
      Function IopSRight (obj: TValue): TNDarray ;

      Function IopAnd    (obj: TValue): TNDarray ;
      Function IopOr     (obj: TValue): TNDarray ;
      Function IopXor    ( obj:TValue): TNDarray ;

      property Items[index: string]:TNDarray           read Getitem write SetItem; default;
      property Items[index: TArray<Integer>] :TNDarray read Getitem write SetItem ;default;
      property Items[index: TArray<TNDarray>]:TNDarray read Getitem write SetItem ;default;
      property Items[index: TArray<TVarRec>] :TNDarray read Getitem write SetItem ;default;
  end;

  TNDArray<T>  = class (TNDArray)
    public
      constructor  Create(Value: TNDarray); overload;
      constructor  Create(_pyobject: PPyObject);overload;

  end;

implementation
  uses System.RegularExpressions,System.Variants,
       np.Base, np.Api,np.Utils;

{ TDtype }

constructor TDtype.Create(_pyobject: PPyObject);
begin
    inherited Create(_pyobject);
end;

constructor TDtype.Create(t: TDtype);
begin
    inherited Create(t.Handle);
end;

{ TFlags }

constructor TFlags.Create(_pyobject: PPyObject);
begin
    inherited Create(_pyobject) ;
end;

{ TMatrix }

constructor TMatrix.Create(_pyobject: PPyObject);
begin
    inherited Create(_pyobject) ;
end;

{ TMemMapMode }

constructor TMemMapMode.Create(_pyobject: PPyObject);
begin
    inherited Create(_pyobject) ;
end;

{ Tnp_Shape }

constructor Tnp_Shape.Create(shape: TArray<Integer>);
begin
    Self       := default(Tnp_Shape);
    Dimensions := shape;
end;

class operator Tnp_Shape.Equal(a, b: Tnp_Shape): Boolean;
var
  i : Integer;
begin
    if Length(a.Dimensions) <>  Length(b.Dimensions) then Exit(False);

    for i := 0 to Length(a.Dimensions) do
     if a.Dimensions[i] <>  b.Dimensions[i] then Exit(False) ;

    Result:= True;
end;

class operator Tnp_Shape.NotEqual(a, b: Tnp_Shape): Boolean;
begin
    Result := not (a = b);
end;

function Tnp_Shape.GetItem(index:Integer): Integer;
begin
    Result := Dimensions[index];
end;

function Tnp_Shape.ToString: string;
var
  i : Integer;
begin
    Result := '';
    for i := 0 to Length(Dimensions)-1 do
      Result := Result + IntToStr(Dimensions[i]) + ',';

    if Length(Result) > 1 then
    begin
      Result := '(' +Result ;
      Result[Length(Result)] := ')'
    end;

end;

{ Tnp_Slice }

function Tnp_Slice.Create(slice_notation: string): Tnp_Slice;
begin
    Result := default(Tnp_Slice);
    Parse(slice_notation);
end;

function Tnp_Slice.Create(_Start, _Stop, _Step: Integer): Tnp_Slice;
begin
    Result := default(Tnp_Slice);
    Result.Start := _start;
    Result.Stop := _stop;
    Result.Step := _step;
end;

function Tnp_Slice.All: Tnp_Slice;
begin
    Result := default(Tnp_Slice);
    Result := Create()
end;

function Tnp_Slice.ParseSlices(multi_slice_notation: string): TArray<Tnp_Slice>;
var
 regEx : TRegEx;
 res   : TArray<String>;
 i     : Integer;
begin
    regEx := TRegEx.Create(',\s*');
    res   := regEx.Split(multi_slice_notation) ;

    Result := [];
    for i := 0 to High(res) do
    begin
        if (res[i] <> '') and (res[i] <> ' ') then
        begin
            Result := Result + [ Create( res[i] ) ]
        end;
    end;
end;

procedure Tnp_Slice.Parse(slice_notation: string);
var
 regEx : TRegEx;
 match : TMatch;
 iStart,
 iStop,
 iStep : Integer;
 start_string,
 stop_string,
 step_string,
 single_pick_string  : string;
begin
    if slice_notation.IsNullOrWhiteSpace(slice_notation) then
       raise Exception.Create('Slice notation expected, got empty string or null');

    regEx := TRegEx.Create('^\s*([+-]?\s*\d+)?\s*:\s*([+-]?\s*\d+)?\s*(:\s*([+-]?\s*\d+)?)?\s*$|^\s*([+-]?\s*\d+)\s*$');

    match := regEx.Match(slice_notation);
    if not match.Success then
      raise Exception.Create('Invalid slice notation');

    start_string := regEx.Replace(match.Groups[1].Value,'\s+','') ;
    stop_string  := regEx.Replace(match.Groups[2].Value,'\s+','') ;
    step_string  := regEx.Replace(match.Groups[4].Value,'\s+','') ;

    single_pick_string :=regEx.Replace(match.Groups[5].Value,'\s+','') ;

    if not slice_notation.IsNullOrWhiteSpace(single_pick_string) then
    begin
        if not Integer.TryParse(single_pick_string,iStart) then
           raise Exception.Create('Invalid value for start: '+ start_string);

        Start := iStart;
        Stop  := Start + 1;
        Step  := 1;
        IsIndex := True;
        Exit;
    end;

    if start_string.IsNullOrWhiteSpace(start_string) then
      Start := null_
    else begin
         if not Integer.TryParse(start_string,iStart) then
           raise Exception.Create('Invalid value for start: '+ start_string);

         Start := iStart;
    end;
    if stop_string.IsNullOrWhiteSpace(stop_string) then
      Stop := null_
    else begin
         if not Integer.TryParse(stop_string,iStop) then
           raise Exception.Create('Invalid value for stop: '+ stop_string);

         Start := iStop;
    end;
    if step_string.IsNullOrWhiteSpace(step_string) then
      Step := 1
    else begin
         if not Integer.TryParse(step_string,iStep) then
           raise Exception.Create('Invalid value for step: '+ step_string);

         Step := iStep;
    end;
end;

function Tnp_Slice.FormatSlices(slices: TArray<Tnp_Slice>): string;
var
 i : Integer;
begin
    Result := '';
    for i := 0 to High(slices) do
     Result := Result + slices[i].ToString + ',';

    Result := Copy(Result ,1,Length(Result)-1);

end;

function Tnp_Slice.GetAbsStart(dim: Integer): Integer;
var
  absStartN, absStart : Integer;
begin
     if Start  < 0  then absStartN := dim + Start
     else                absStartN := Start;

     if Step  < 0  then
     begin
         if   absStartN = NULL_ then absStart := dim
         else                        absStart := absStartN;
     end else
     begin
         if   absStartN = NULL_ then absStart := 0
         else                        absStart := absStartN;
     end;

     Result := absStart;
end;

function Tnp_Slice.GetAbsStep: Integer;
begin
    Result := Abs(Step);
end;

function Tnp_Slice.GetAbsStop(dim: Integer): Integer;
var
  absStopN, absStop : Integer;
begin
     if Stop  < 0  then absStopN := dim + Stop
     else               absStopN := Stop;

     if Step  < 0  then
     begin
         if   absStopN = NULL_ then absStop := 0
         else                       absStop := absStopN;
     end else
     begin
         if   absStopN = NULL_ then absStop := dim
         else                       absStop := absStopN;
     end;

     Result := absStop;

end;

function Tnp_Slice.GetSize(dim: Integer): Integer;
var
  absStart, absStop,absStep : Integer;
begin
    absStart := GetAbsStart(dim);
    absStop  := GetAbsStop(dim);
    absStep  := GetAbsStep;

    Result := ((absStop - absStart)+(absStep-1)) div absStep;
end;

function Tnp_Slice.Index(index: Integer): Tnp_Slice;
begin
    Create(index,index+1);
    IsIndex := True;
end;

class operator Tnp_Slice.NotEqual(a, b: Tnp_Slice): Boolean;
begin
    Result := not(a = b)
end;

class operator Tnp_Slice.Equal(a, b: Tnp_Slice): Boolean;
begin
    Result := (a.Start = b.Start) and (a.Stop = b.Stop) and (a.Step = b.Step)
end;

function Tnp_Slice.ToPython: TPythonObject;
var
  sEval : string;

begin
    { TODO -oMax -c : modificare call python engine 26/01/2020 16:59:16 }
    sEval := Format('slice(%s,%s,%s)',[FormatNullableIntForPython(Start),FormatNullableIntForPython(Stop),FormatNullableIntForPython(Step)]);

    Result := TPythonObject.Create( g_MyPyEngine.EvalString(sEval) );
end;

function Tnp_Slice.FormatNullableIntForPython(i: Integer): string;
begin
    if i = null_ then
      Result := 'None'
    else
      Result := IntToStr(i)
end;

function Tnp_Slice.ToString: string;
var
  sStart,
  sStop  : string;
  opt_step : string;
begin
    if IsIndex then
    begin
      if   Start = NULL_ then Start := 0;
      Exit(IntToStr(Start));
    end;
    if Step = 1 then opt_step := ''
    else             opt_step := ':'+ IntToStr(Step) ;

    if Start = 0 then sStart := ''
    else              sStart := IntToStr(Start) ;

    if Stop = NULL_ then  sStop := ''
    else                  sStop := IntToStr(Stop);

    Result := sStart +':'+ sStop + opt_step;
end;

{ TNDArray }

constructor TNDArray.Create(_pyobject: PPyObject);
begin
    inherited Create(_pyobject);
end;

constructor TNDArray.Create(t: TNDArray);
begin
    inherited Create(t.Handle);
end;

//------------------------------
// Comparison operators:
//------------------------------
function TNDArray.equal(obj: TValue): TNDarray<Boolean>;
begin
    Result := TNDarray<Boolean>.Create( InvokeMethod('__eq__', [TNumPy.ToPython(obj)] ) );
end;

function TNDArray.not_equals(obj: TValue): TNDarray<Boolean>;
begin
   Result := TNDarray<Boolean>.Create( InvokeMethod('__ne__', [TNumPy.ToPython(obj)] ) );
end;

class function TNDArray.opLess(a: TNDarray; obj: TValue): TNDArray<Boolean>;
begin
     Result := TNDarray<Boolean>.Create( a.InvokeMethod('__lt__', [TNumPy.ToPython(obj)] ) );
end;

class function TNDArray.opLessEq(a: TNDarray; obj: TValue): TNDArray<Boolean>;
begin
    Result := TNDarray<Boolean>.Create( a.InvokeMethod('__le__', [TNumPy.ToPython(obj)] ) );
end;

class function TNDArray.opGreat(a: TNDarray; obj: TValue): TNDArray<Boolean>;
begin
    Result := TNDarray<Boolean>.Create( a.InvokeMethod('__gt__', [TNumPy.ToPython(obj)] ) );
end;

class function TNDArray.opGreatEq(a: TNDarray; obj: TValue): TNDArray<Boolean>;
begin
    Result := TNDarray<Boolean>.Create( a.InvokeMethod('__ge__', [TNumPy.ToPython(obj)] ) );
end;

//------------------------------
// Truth value of an array(bool) :
//------------------------------
class function TNDArray.nonzero(a: TNDarray): TNDarray<Boolean>;
begin
    Result := TNDarray<Boolean>.Create( a.InvokeMethod('__nonzero__',[]) );
end;


//------------------------------
// Unary operations:
//------------------------------
class function TNDArray.opNeg(a: TNDarray): TNDarray;
begin
   Result := TNDarray.Create( a.InvokeMethod('__neg__',[]) );
end;

class function TNDArray.opNot(a: TNDarray): TNDarray;
begin
    Result := TNDarray.Create( a.InvokeMethod('__invert__',[]) );
end;

class function TNDArray.opPos(a: TNDarray): TNDarray;
begin
    Result := TNDarray.Create( a.InvokeMethod('__pos__',[]) );
end;

//------------------------------
// Arithmetic operators:
//------------------------------
class function TNDArray.opSLeft(a: TNDarray; obj: integer): TNDarray;
begin
    Result := TNDarray.Create( a.InvokeMethod('__lshift__', [TNumPy.ToPython(obj)] ) );
end;

class function TNDArray.opSRight(a: TNDarray; obj: integer): TNDarray;
begin
    Result := TNDarray.Create( a.InvokeMethod('__rshift__', [TNumPy.ToPython(obj)] ) );
end;

class function TNDArray.opOr(a: TNDarray; obj: integer): TNDarray;
begin
    Result := TNDarray.Create( a.InvokeMethod('__or__', [TNumPy.ToPython(obj)] ) );
end;

class function TNDArray.opXor(a: TNDarray; obj: integer): TNDarray;
begin
    Result := TNDarray.Create( a.InvokeMethod('__xor__', [TNumPy.ToPython(obj)] ) );
end;

function TNDArray.opDivMod(obj: TValue): TNDarray;
begin
     Result := TNDarray.Create( InvokeMethod('__divmod__', [TNumPy.ToPython(obj)] ) );
end;

function TNDArray.opPow(obj: TValue): TNDarray;
begin
    Result := TNDarray.Create( InvokeMethod('__pow__', [TNumPy.ToPython(obj)] ) );
end;

class function TNDArray.opSub(a: TNDarray; obj: TValue): TNDarray;
begin
    Result := TNDarray.Create( a.InvokeMethod('__sub__', [TNumPy.ToPython(obj)] ) );
end;

class function TNDArray.opSub(a, obj: TNDarray): TNDarray;
begin
    Result := TNDarray.Create( a.InvokeMethod('__sub__', [TNumPy.ToPython(obj)] ) );
end;

class function TNDArray.opAdd(a: TNDarray; obj: TValue): TNDarray;
begin
    Result := TNDarray.Create( a.InvokeMethod('__add__', [TNumPy.ToPython(obj)] ) );
end;

class function TNDArray.opAdd(obj: TValue; a: TNDarray): TNDarray;
begin
    Result := TNDarray.Create( a.InvokeMethod('__add__', [TNumPy.ToPython(obj)] ) );
end;

class function TNDArray.opAdd(a, obj: TNDarray): TNDarray;
begin
    Result := TNDarray.Create( a.InvokeMethod('__add__', [TNumPy.ToPython(obj)] ) );
end;

class function TNDArray.opAnd(a: TNDarray; obj: integer): TNDarray;
begin
    Result := TNDarray.Create( a.InvokeMethod('__and__', [TNumPy.ToPython(obj)] ) );
end;

class function TNDArray.opDiv(a: TNDarray; obj: TValue): TNDarray;
begin
    Result := TNDarray.Create( a.InvokeMethod('__truediv__', [TNumPy.ToPython(obj)] ) );
end;

class function TNDArray.opDiv(a, obj: TNDarray): TNDarray;
begin
    Result := TNDarray.Create( a.InvokeMethod('__truediv__', [TNumPy.ToPython(obj)] ) );
end;

class function TNDArray.floordiv(a: TNDarray; obj: TValue): TNDarray;
begin
   Result := TNDarray.Create( a.InvokeMethod('__floordiv__', [TNumPy.ToPython(obj)] ) );
end;

class function TNDArray.opMod(a: TNDarray; obj: TValue): TNDarray;
begin
   Result := TNDarray.Create( a.InvokeMethod('__mod__', [TNumPy.ToPython(obj)] ) );
end;

class function TNDArray.opMul(a, obj: TNDarray): TNDarray;
begin
    Result := TNDarray.Create( a.InvokeMethod('__mul__', [TNumPy.ToPython(obj)] ) );
end;

class function TNDArray.opMul(a: TNDarray; obj: TValue): TNDarray;
begin
    Result := TNDarray.Create( a.InvokeMethod('__mul__', [TNumPy.ToPython(obj)] ) );
end;

class function TNDArray.opMul(obj: TValue; a: TNDarray): TNDarray;
begin
    Result := TNDarray.Create( a.InvokeMethod('__mul__', [TNumPy.ToPython(obj)] ) );
end;

//------------------------------
// Arithmetic, in-place:
//------------------------------
function TNDArray.opIAdd(obj: TValue): TNDarray;
begin
    Result := TNDarray.Create( InvokeMethod('__iadd__', [TNumPy.ToPython(obj)] ) );
end;

function TNDArray.opISub(obj: TValue): TNDarray;
begin
    Result := TNDarray.Create( InvokeMethod('__isub__', [TNumPy.ToPython(obj)] ) );
end;

function TNDArray.opIMul(obj: TValue): TNDarray;
begin
    Result := TNDarray.Create( InvokeMethod('__imul__', [TNumPy.ToPython(obj)] ) );
end;

function TNDArray.opITrueDiv(obj: TValue): TNDarray;
begin
    Result := TNDarray.Create( InvokeMethod('__itruediv__', [TNumPy.ToPython(obj)] ) );
end;

function TNDArray.IFloorDiv(obj: TValue): TNDarray;
begin
   Result := TNDarray.Create( InvokeMethod('__floordiv__', [TNumPy.ToPython(obj)] ) );
end;

function TNDArray.opIMod(obj: TValue): TNDarray;
begin
    Result := TNDarray.Create( InvokeMethod('__imod__', [TNumPy.ToPython(obj)] ) );
end;

function TNDArray.opIPow(obj: TValue): TNDarray;
begin
    Result := TNDarray.Create( InvokeMethod('__ipow__', [TNumPy.ToPython(obj)] ) );
end;

function TNDArray.IopSLeft(obj: TValue): TNDarray;
begin
   Result := TNDarray.Create( InvokeMethod('__ilshift__', [TNumPy.ToPython(obj)] ) );
end;

function TNDArray.IopSRight(obj: TValue): TNDarray;
begin
   Result := TNDarray.Create( InvokeMethod('__irshift__', [TNumPy.ToPython(obj)] ) );
end;

function TNDArray.IopAnd(obj: TValue): TNDarray;
begin
    Result := TNDarray.Create( InvokeMethod('__iand__', [TNumPy.ToPython(obj)] ) );
end;

function TNDArray.IopOr(obj: TValue): TNDarray;
begin
    Result := TNDarray.Create( InvokeMethod('__ior__', [TNumPy.ToPython(obj)] ) );
end;

function TNDArray.IopXor(obj: TValue): TNDarray;
begin
   Result := TNDarray.Create( InvokeMethod('__ixor__', [TNumPy.ToPython(obj)] ) );
end;

function TNDArray.opIDiv(obj: TValue): TNDarray;
begin
    Result := TNDarray.Create( InvokeMethod('__idiv__', [TNumPy.ToPython(obj)] ) );
end;


function TNDArray.Getitem(index: string): TNDarray;
var
  tuple   : TPyTuple;
  slice   : Tnp_Slice;
  aSlice  : TArray<Tnp_Slice>;
  resSlice: TArray<TPythonObject>;
  pyObj   : TPythonObject;
  i       : Integer;
begin
    aSlice := slice.ParseSlices(index) ;

    for i := 0 to length(aSlice) -1 do
    begin
        slice := aSlice[i];

        if slice.IsIndex then   pyObj := TPyInt.Create(slice.Start)
        else                    pyObj := slice.ToPython ;

        resSlice := resSlice + [ pyObj ];
    end;
    tuple := TPyTuple.Create( resSlice );

    Result :=  TNDArray.Create( TPythonObject(Self)[tuple] );
end;

procedure TNDArray.SetItem(index: string; const Value: TNDarray);
var
  tuple   : TPyTuple;
  slice   : Tnp_Slice;
  aSlice  : TArray<Tnp_Slice>;
  resSlice: TArray<TPythonObject>;
  pyObj   : TPythonObject;
  i       : Integer;
begin
    aSlice := slice.ParseSlices(index) ;

    for i := 0 to length(aSlice) -1 do
    begin
        slice := aSlice[i];

        if slice.IsIndex then   pyObj := TPyInt.Create(slice.Start)
        else                    pyObj := slice.ToPython ;

        resSlice := resSlice + [ pyObj ];
    end;
    tuple := TPyTuple.Create( resSlice );

    TPythonObject(Self)[tuple] := TNumPy.ToPython(TPythonObject(value) )

end;

function TNDArray.Getitem(index: TArray<Integer>): TNDarray;
var
 tuple : TPyTuple;

  res : Integer;
begin
    tuple  := TNumPy.ToTuple(TValue.ArrayOfToValueArray<Integer>(index));
    Result := TNDarray.Create( TPythonObject(Self)[tuple] );
end;

procedure TNDArray.SetItem(index: TArray<Integer>; const Value: TNDarray);
var
 tuple : TPyTuple;
begin
    tuple  := TNumPy.ToTuple(TValue.ArrayOfToValueArray<Integer>(index));
    TPythonObject(Self)[tuple] := TNumPy.ToPython(value );
end;

function TNDArray.Getitem(index: TArray<TNDarray>): TNDarray;
var
 aObj  : TArray<TPythonObject>;
 tuple : TPyTuple;
 i     : Integer;
begin
    for i := 0 to High(index) do
      aObj := aObj + [ index[i] ];

    tuple := TPyTuple.Create(aObj);

    Result := TNDarray.Create( TPythonObject(Self)[tuple] );
end;

procedure TNDArray.SetItem(index: TArray<TNDarray>; const Value: TNDarray);
var
 aObj  : TArray<TPythonObject>;
 tuple : TPyTuple;
 i     : Integer;
begin
    for i := 0 to High(index) do
      aObj := aObj + [ index[i] ];

    tuple := TPyTuple.Create(aObj);

    TPythonObject(Self)[tuple] := TNumPy.ToPython(TPythonObject(value) )
end;


function TNDArray.Getitem(index: TArray<TVarRec>): TNDarray;
var
  tuple   : TPyTuple;
  item    : TVarRec;
  slice   : Tnp_Slice;
  res     : TArray<TPythonObject>;
  pyObj   : TPythonObject;
  i       : Integer;
begin
    pyObj := nil;
    for i := 0 to length(index) -1 do
    begin
        item := index[i];
        case item.VType of
         vtInteger    :  pyObj := TPyInt.Create( index[i].VInteger ) ;
         vtString     :  pyObj := slice.Create( string(index[i].VString^) ).ToPython ;
         vtAnsiString :  pyObj := slice.Create( string(index[i].VAnsiString^) ).ToPython ;
         vtClass      :
          begin
              if LowerCase(item.VClass.ClassName) = 'tndarray' then
                 pyObj := TPythonObject.Create(  TNDArray(index).Handle );
          end;
         else
            pyObj := TNumPy.ToPython(item) ;
        end;

        res := res + [ pyObj ];
    end;
    tuple := TPyTuple.Create( res );

    Result :=  TNDArray.Create( TPythonObject(Self)[tuple] );

end;

procedure TNDArray.SetItem(index: TArray<TVarRec>; const Value: TNDarray);
var
  tuple   : TPyTuple;
  item    : TVarRec;
  slice   : Tnp_Slice;
  res     : TArray<TPythonObject>;
  pyObj   : TPythonObject;
  i       : Integer;
begin
    pyObj := nil;
    for i := 0 to length(index) -1 do
    begin
        item := index[i];
        case item.VType of
         vtInteger    :  pyObj := TPyInt.Create( index[i].VInteger ) ;
         vtString     :  pyObj := slice.Create( string(index[i].VString^) ).ToPython ;
         vtAnsiString :  pyObj := slice.Create( string(index[i].VAnsiString^) ).ToPython ;
         vtClass      :
          begin
              if LowerCase(item.VClass.ClassName) = 'tndarray' then
                 pyObj := TPythonObject.Create(  TNDArray(index).Handle );
          end;
         else
            pyObj := TNumPy.ToPython(item) ;
        end;

        res := res + [ pyObj ];
    end;
    tuple := TPyTuple.Create( res );

    TPythonObject(Self)[tuple] := TNumPy.ToPython(TPythonObject(value) )

end;

procedure TNDArray.itemset(args: TArray<TValue>);
var
  pyargs : TPyTuple;
  kwargs : TPyDict;

begin
    pyargs := TNumPy.ToTuple(args);
    kwargs := TPyDict.Create;

    InvokeMethod('itemset', pyargs, kwargs);

end;

function TNDArray.ctypes: PPyObject;
begin
    Result := GetAttr('ctypes').Handle
end;

function TNDArray.data: PPyObject;
begin
    Result := GetAttr('data').Handle
end;

function TNDArray.dtype: TDtype;
begin
    Result := TDtype.Create( GetAttr('dtype') );
end;

function TNDArray.flags: TFlags;
begin
    Result := TFlags.Create( GetAttr('flags').Handle );
end;

function TNDArray.flat: PPyObject;
begin
    Result := TFlags.Create( GetAttr('flat').Handle ).Handle;
end;

function TNDArray.itemsize: Integer;
begin
    Result := GetAttr('itemsize').AsInteger;
end;

function TNDArray.len: Integer;
begin
    Result := InvokeMethod('__len__',[]).AsInteger;
end;

function TNDArray.nbytes: Integer;
begin
    Result := GetAttr('nbytes').AsInteger
end;

function TNDArray.ndim: Integer;
begin
    Result := GetAttr('ndim').AsInteger
end;

function TNDArray.repr: string;
begin
    Result := InvokeMethod('__repr__',[]).AsString;
end;

{ DONE -oMax -c : aggiornare con reshape 08/02/2020 21:17:15 }
function TNDArray.reshape(newshape: TArray<Integer>): TNDarray;
begin
    Result := TNumPy.reshape(Self, Tnp_Shape.Create(newshape));
end;

procedure TNDArray.resize(new_shape: Tnp_Shape; refcheck: PBoolean);
var
  pyargs : TPyTuple;
  kwargs : TPyDict;

begin
    pyargs := TNumPy.ToTuple([TValue.FromShape(new_shape)]);
    kwargs := TPyDict.Create;

    if (refcheck <> nil) then  kwargs['refcheck'] := TNumPy.ToPython(refcheck^);
    InvokeMethod('resize', pyargs, kwargs);
end;

function TNDArray.shape: Tnp_Shape;
var
  v   : TArray<Integer>;
begin
    v := GetAttr('shape').AsArrayofInt;

    Result := Tnp_Shape.Create(v) ;
end;

function TNDArray.size: Integer;
begin
    Result := GetAttr('size').AsInteger
end;

function TNDArray.str: string;
begin
    Result :=InvokeMethod('__str__',[]).AsString;
end;

function TNDArray.strides: TArray<Integer>;
begin
     Result := GetAttr('strides').AsArrayofInt
end;

function TNDArray.T: TNDArray;
begin
    Result := TNDArray.Create( GetAttr('T') )
end;

function TNDArray.tobytes(order: PChar): TArray<Byte>;
var
  pyargs : TPyTuple;
  kwargs : TPyDict;
  py     : TPythonObject;

begin
    raise Exception.Create('TODO: this needs to be implemented with Marshal.Copy');

    pyargs := TNumPy.ToTuple([]);
    kwargs := TPyDict.Create;

    if (order <> nil) then  kwargs['order'] := TNumPy.ToPython(order^);
    py :=InvokeMethod('tobytes', pyargs, kwargs);

    Result := TNumPy.ToCsharp<TArray<byte>>(py)
end;

function TNDArray.tostring(order: PChar): TArray<Byte>;
begin
    Result := tobytes(order)
end;

procedure TNDArray.view(dtype: TDtype; tipo: Variant);
var
  pyargs : TPyTuple;
  kwargs : TPyDict;

begin
    raise Exception.Create('Get python type "ndarray" and "matrix" and substitute them for the given .NET type');


    pyargs := TNumPy.ToTuple([]);
    kwargs := TPyDict.Create;

    if (dtype <> nil)          then  kwargs['dtype'] := TNumPy.ToPython(dtype);
    if ( not VarIsNull(tipo) ) then  kwargs['tipo']  := TNumPy.ToPython(TValue.FromVariant(tipo));
    InvokeMethod('view', pyargs, kwargs);

end;

function TNDArray.base: TNDArray;
var
 py : TPythonObject;
begin
     py := GetAttr('base');

     if py.Handle = nil then Exit(nil);

     Result := TNDArray.Create( py )
end;

function TNDArray.asscalar<T>: T;
begin
    Result := TNumPy.asscalar<T>(self);
end;

{ TNDArray<T> }

constructor TNDArray<T>.Create(_pyobject: PPyObject);
begin
    inherited Create(_pyobject);
end;

constructor TNDArray<T>.Create(Value: TNDarray);
begin
    inherited Create(Value);
end;

end.
