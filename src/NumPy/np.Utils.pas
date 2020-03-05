unit np.Utils;

interface
  uses Winapi.Windows, System.SysUtils,System.Rtti,
       Python.Utils,
       np.Models;

type

 TArray2D<T> = array of TArray<T>;
 TArray3D<T> = array of TArray2D<T>;

 TValueHelper = record helper for TValue
   public
    function AsShape     : Tnp_Shape;
    function AsSlice     : Tnp_Slice;
    function AsNDArray   : TNDArray; overload;
    function AsNDArray<T>: TNDArray<T>; overload;
    function AsArray     : TArray<TValue>;

    class function FromArray<T>(Params: TArray<T>): TValue; static;
    class function ArrayOfToValueArray<T>(Params: TArray<T>): TArray<TValue>; static;
    class function FromShape(Params: Tnp_Shape): TValue; static;
    class function FromSlice(Params: Tnp_Slice): TValue; static;

  end;

  TTupleSolver = class
  private
       class procedure GetNdListFromTuple(obj: TPythonObject; var res: TArray<TNDArray>);
       class procedure GetTTupleList<T>(obj: TPythonObject; var res: TArray<T>); static;
    public
       class function TupleToList<T>(obj: TPythonObject): TArray<T>; overload;
       class function TupleToList(obj: TPythonObject): TArray<TNDArray>; overload;
  end;

implementation
      uses np.Base;

{ TValueHelper }

class function TValueHelper.ArrayOfToValueArray<T>(Params: TArray<T>): TArray<TValue>;
var
  i: Integer;
begin

  for i := 0 to Length(Params) -1 do
    Result := Result + [ TValue.From<T>(Params[i]) ];

end;

class function TValueHelper.FromShape(Params: Tnp_Shape): TValue;
begin
    Result := TValue.From<Tnp_Shape>(Params);
end;

class function TValueHelper.FromSlice(Params: Tnp_Slice): TValue;
begin
    Result := TValue.From<Tnp_Slice>(Params);
end;

class function TValueHelper.FromArray<T>(Params: TArray<T>): TValue;
begin
    Result := TValue.From<TArray<T>>(Params);
end;

function TValueHelper.AsArray: TArray<TValue>;
var
  i: Integer;
begin
   Result := [];
   for i := 0 to Self.GetArrayLength -1 do
     Result := Result + [ self.GetArrayElement(i) ];
end;

function TValueHelper.AsNDArray: TNDArray;
begin
    Result := Self.AsType<TNDArray>;
end;

function TValueHelper.AsNDArray<T>: TNDArray<T>;
begin
    Result := Self.AsType<TNDArray<T>>;
end;

function TValueHelper.AsShape: Tnp_Shape;
begin
    Result := Self.AsType<Tnp_Shape>;
end;

function TValueHelper.AsSlice: Tnp_Slice;
begin
    Result := Self.AsType<Tnp_Slice>;
end;

{ TTupleSolver }

class function TTupleSolver.TupleToList(obj: TPythonObject): TArray<TNDArray>;
var
  Iter: TPyIter;
begin
    Iter :=  TPyIter.Create(obj);

    GetNdListFromTuple(Iter,Result);
end;

class procedure TTupleSolver.GetNdListFromTuple(obj: TPythonObject; var res: TArray<TNDArray>);
var
  Iter: TPyIter;
  r   : TPythonObject;
begin
    Iter :=  TPyIter.Create(obj);

    while Iter.MoveNext do
    begin
        r := TPythonObject(Iter.Current);
        if TPyTuple.IsTupleType(r) then
        begin
            GetNdListFromTuple(r,res);
            Continue;
        end;

        res := res + [ TNDArray.Create(r)  ];
    end;

end;

class function TTupleSolver.TupleToList<T>(obj: TPythonObject): TArray<T>;
var
  Iter: TPyIter;
begin
    Iter :=  TPyIter.Create(obj);

    GetTTupleList<T>(Iter,Result);

end;

class procedure TTupleSolver.GetTTupleList<T>(obj: TPythonObject; var res: TArray<T>);
var
  Iter: TPyIter;
  r   : TPythonObject;
begin
    Iter :=  TPyIter.Create(obj);

    while Iter.MoveNext do
    begin
        r := TPythonObject(Iter.Current);
        if TPyTuple.IsTupleType(r) then
        begin
            GetTTupleList<T>(r,res);
            Continue;
        end;

        res := res + [ TNumPy.ToCsharp<T>(r)  ];
    end;

end;

end.
