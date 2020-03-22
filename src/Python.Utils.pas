{*******************************************************}
{                                                       }
{       Numpy Wrapper for Delphi                        }
{                                                       }
{       Copyright (C) 2020 Pigrecos(Max)                }
{                                                       }
{*******************************************************}
unit Python.Utils;

//PythonObject.cs
//PythonObject.gen.cs


interface
   uses Winapi.Windows,System.SysUtils,System.Generics.Collections,
        PythonEngine,
        PythonGUIInputOutput;

type
 TPyType = (ptNull,ptObject, ptString, ptInt16, ptInt32, ptInt64, ptSingle, ptDouble , mtDecimal, mtBoolean);

 // constants.cs
 Constants = (inf, neg_inf);

 TPyTuple = class;
 TPyDict  = class;

 // PythonObject.cs / PythonObject.gen.cs
 TPythonObject = class
   private
     FHandle  : PPyObject;

     function  GetItem(key:   TPythonObject ) : TPythonObject; overload;
     function  GetItem(key:   string)         : TPythonObject; overload;
     function  GetItem(index: Integer)        : TPythonObject; overload;

     procedure  SetItem(key:   TPythonObject;  value: TPythonObject ) ; overload ;
     procedure  SetItem(key:   string;         value: TPythonObject ) ; overload ;
     procedure  SetItem(index: Integer;        value: TPythonObject ) ; overload ;

   public
     constructor Create(t: TPythonObject); overload;
     constructor Create(_pyobject: PPyObject); overload;
     constructor Create; overload;
     destructor Destroy; override;

     function IsNone: Boolean;
     function AsInteger: Integer;
     function AsString:  string;
     function AsDouble:  double;
     function AsBoolean: boolean;
     function AsArrayofString: TArray<String>;
     function AsArrayofInt: TArray<Integer>;
     function AsArrayofDouble: TArray<Double>;
     function AsArrayofPyObj: TArray<TPythonObject>;

     function ToString: string;

     function  GetAttr(name: string):TPythonObject;
     procedure SetAttr(const varName: AnsiString; value: TPythonObject);

     class function ImportModule( Name: string):TPythonObject;
     class function ModuleFromString(name, code: string): TPythonObject; static;
     class function None: TPythonObject;

     function Invoke(args: TArray<TPythonObject>):TPythonObject; overload;
     function Invoke(args: TPyTuple):TPythonObject; overload;
     function Invoke(args: TArray<TPythonObject>; kw: TPyDict ):TPythonObject; overload;
     function Invoke(args: TPyTuple; kw: TPyDict ):TPythonObject; overload;

     function InvokeMethod(name: string; args: TArray<TPythonObject> = []):TPythonObject; overload;
     function InvokeMethod(name: string; args: TPyTuple):TPythonObject; overload;
     function InvokeMethod(name: string; args: TArray<TPythonObject>; kw: TPyDict):TPythonObject; overload;
     function InvokeMethod(name: string; args: TPyTuple; kw: TPyDict ):TPythonObject; overload;

     property Handle                        : PPyObject    read FHandle;
     property ItemFrom[index :TPythonObject]:TPythonObject read GetItem write SetItem; default;
     property Itemfrom[index :string]       :TPythonObject read GetItem write SetItem; default;
     property ItemFrom[index :Integer]      :TPythonObject read GetItem write SetItem; default;

 end;

 TPyNumber = class(TPythonObject)
  public
    constructor Create(_pyobject: PPyObject); overload;
    constructor Create; overload;

    function IsNumberType(value: TPythonObject): Boolean;
 end;

 TPyDict = class(TPythonObject)
  public
    constructor Create(_pyobject: PPyObject); overload;
    constructor Create(o: TPythonObject); overload;
    constructor Create; overload;

    function IsDictType(value: TPythonObject) : Boolean;
    function HasKey(key: TPythonObject) : Boolean; overload;
    function HasKey(key: string) : Boolean; overload;
    function Keys : TPythonObject;
    function Values : TPythonObject;
    function Items : TPythonObject;
    function Copy : TPyDict;
    procedure Update(other: TPythonObject);
    procedure Clear;
 end;

 TPyLong = class(TPyNumber)
  public
    constructor Create(_pyobject: PPyObject); overload;
    constructor Create(o: TPythonObject); overload;
    constructor Create(value: Integer); overload;
    constructor Create(value: Cardinal); overload;
    constructor Create(value: Int64); overload;
    constructor Create(value: UInt64); overload;
    constructor Create(value: Int16); overload;
    constructor Create(value: UInt16); overload;
    constructor Create(value: Int8); overload;
    constructor Create(value: UInt8); overload;
    constructor Create(value: Double); overload;
    constructor Create(value: string); overload;
    function  IsLongType(value: TPythonObject):Boolean;
    function  AsLong(value: TPythonObject): TPyLong;

    function ToInt16: Int16;
    function ToInt32: Int32;
    function ToInt64: Int64;
 end;

 TPyInt = class(TPyNumber)
  public
    constructor Create(_pyobject: PPyObject); overload;
    constructor Create(o: TPythonObject); overload;
    constructor Create(value: Integer); overload;
    constructor Create(value: Cardinal); overload;
    constructor Create(value: Int64); overload;
    constructor Create(value: UInt64); overload;
    constructor Create(value: Int16); overload;
    constructor Create(value: UInt16); overload;
    constructor Create(value: Int8); overload;
    constructor Create(value: UInt8); overload;
    constructor Create(value: string); overload;
    function  IsIntType(value: TPythonObject):Boolean;
    function  AsInt(value: TPythonObject): TPyInt;

    function ToInt16: Int16;
    function ToInt32: Int32;
    function ToInt64: Int64;
 end;

 TPyFloat = class(TPyNumber)
  public
    constructor Create(_pyobject: PPyObject); overload;
    constructor Create(o: TPythonObject); overload;
    constructor Create(value: Double); overload;
    constructor Create(value: string); overload;
    function  IsFloatType(value: TPythonObject):Boolean;
    function  AsFloat(value: TPythonObject): TPyFloat;
 end;

 TPySequence = class(TPythonObject)
  public
    constructor Create(_pyobject: PPyObject); overload;
    constructor Create; overload;
    function  IsSequenceType(value: TPythonObject) : Boolean;
    function  GetSlice(i1, i2: Integer): TPythonObject;
    procedure SetSlice(i1, i2: Integer; v: TPythonObject);
    procedure DelSlice(i1, i2: Integer);
    function  Index(item: TPythonObject): Integer;
    function  Contains(item: TPythonObject): Boolean;
    function  Concat(other: TPythonObject): TPythonObject;
    function  Repeatt(count: Integer): TPythonObject;
 end;

 TPyString = class(TPySequence)
  public
    constructor Create(_pyobject: PPyObject); overload;
    constructor Create(o: TPythonObject); overload;
    constructor Create(s: string); overload;

    function IsStringType(value: TPythonObject) : Boolean;
 end;

 TPyAnsiString = class(TPySequence)
  public
    constructor Create(_pyobject: PPyObject); overload;
    constructor Create(o: TPythonObject); overload;
    constructor Create(s: AnsiString ); overload;

    function IsStringType(value: TPythonObject) : Boolean;
 end;

 TPyTuple = class(TPySequence)
  public
    constructor Create; overload;
    constructor Create(_pyobject: PPyObject); overload;
    constructor Create(o: TPythonObject); overload;
    constructor Create(const objects : array of PPyObject); overload;
    constructor Create(const objects: array of const);overload;
    constructor Create(const objects: TArray<TPythonObject>); overload;

    class function IsTupleType(value: TPythonObject): Boolean;
    function AsTuple(value: TPythonObject): TPyTuple;
 end;

 TPyList = class(TPySequence)
  public
    constructor Create; overload;
    constructor Create(_pyobject: PPyObject); overload;
    constructor Create(o: TPythonObject); overload;
    constructor Create(const objects : array of PPyObject); overload;
    constructor Create(const objects: array of TPythonObject); overload;

    function  IsListType(value: TPythonObject): Boolean;
    function  AsList(value: TPythonObject): TPyList;
    procedure Append(item: TPythonObject) ;
    procedure Insert(index : Integer ;item: TPythonObject) ;
    procedure Reverse;
    procedure Sort;
 end;

 TPyIter = class(TPythonObject,IEnumerator)
  private
    FCurrent : TPythonObject;
    function GetCurrent: TObject;
  public
    constructor Create(_pyobject: PPyObject); overload;
    constructor Create(iterable: TPythonObject); overload;
    function QueryInterface(const IID: TGUID; out Obj): HResult; stdcall;
    function _AddRef: Integer; stdcall;
    function _Release: Integer; stdcall;
    function MoveNext: Boolean;
    procedure Reset;

    property Current: TObject read GetCurrent;

 end;

 procedure InitGlobal(PyGuiIO: TPythonGUIInputOutput = nil);

 var
   g_MyPyEngine : TPythonEngine;

implementation
        uses
           System.TypInfo,
           System.Variants;

{ TPythonObject }

procedure InitGlobal(PyGuiIO: TPythonGUIInputOutput);
begin
    if g_MyPyEngine = nil then
    begin
      g_MyPyEngine := TPythonEngine.Create(nil);
      g_MyPyEngine.IO := PyGuiIO;

      g_MyPyEngine.LoadDll;
    end;

end;

constructor TPythonObject.Create;
begin
    if g_MyPyEngine = nil then  InitGlobal;
end;

constructor TPythonObject.Create(_pyobject: PPyObject);
begin
     create;
     FHandle := _pyobject;
end;

destructor TPythonObject.Destroy;
begin
    if FHandle <> nil then
      g_MyPyEngine.Py_XDECREF(FHandle);

    g_MyPyEngine.PyErr_Clear;
end;

constructor TPythonObject.Create(t: TPythonObject);
begin
    create;
    FHandle := t.Handle;
end;

// warning, this function will increase the refcount of value,
// so, if you don't want to keep a link, don't forget to decrement
// the refcount after the SetVar method.
procedure TPythonObject.SetAttr( const varName : AnsiString; value : TPythonObject );
begin
   if Assigned(g_MyPyEngine) and Assigned( FHandle ) then
   begin
      if g_MyPyEngine.PyObject_SetAttrString(FHandle, PAnsiChar(varName), value.Handle ) <> 0 then
        raise EPythonError.CreateFmt( 'Could not set var "%s" in module "%s"', [varName] );
      g_MyPyEngine.Py_XDecRef( value.Handle);
   end
   else
     raise EPythonError.CreateFmt( 'Can''t set var "%s" , because it is not yet initialized', [varName] );
end;

// warning, this function will increase the refcount of value,
// so, if you don't want to keep a link, don't forget to decrement
// the refcount after the GetVar method.
function TPythonObject.GetAttr(name: string): TPythonObject;
var
   attr: PPyObject;
begin
    if Assigned(g_MyPyEngine) and Assigned( FHandle ) then
    begin
        attr := g_MyPyEngine.PyObject_GetAttrString(FHandle, PAnsiChar(AnsiString(name)));
        g_MyPyEngine.PyErr_Clear;
        if attr = nil then
          raise EPythonError.Create('GetAttr: can''t get Attribute : '+ name);

        Result := TPythonObject.Create(attr);
        //g_MyPyEngine.Py_XDecRef(Result.Handle); // keep a borrowed reference.
    end
    else
      raise EPythonError.CreateFmt( 'Could not get var "%s" , because it is not yet initialized', [name] );
end;

class function TPythonObject.ModuleFromString(name:string; code:string): TPythonObject;
var
  c,m : PPyObject;
begin
    c := g_MyPyEngine.Py_CompileString(PAnsiChar(AnsiString(code)), 'none', file_input);
    g_MyPyEngine.CheckError ;
    m := g_MyPyEngine.PyImport_ExecCodeModule(AnsiString(name), c);
    g_MyPyEngine.CheckError ;

    Result := TPythonObject.Create(m);
end;

function TPythonObject.IsNone: Boolean;
begin
    Result :=  Handle = g_MyPyEngine.Py_None;
end;

class function TPythonObject.None: TPythonObject;
begin
    Result := TPythonObject.Create( g_MyPyEngine.ReturnNone )
end;

class function TPythonObject.ImportModule(Name: string): TPythonObject;
var
  res : PPyObject;
begin
    res := g_MyPyEngine.PyImport_ImportModule(PAnsiChar(AnsiString(Name)) ) ;
    if g_MyPyEngine.PyErr_Occurred <> nil then
    begin
        g_MyPyEngine.PyErr_Clear;
        Exit(nil);
    end;
    Result := TPythonObject.Create(res);
end;

function TPythonObject.Invoke(args: TArray<TPythonObject>): TPythonObject;
var
  t   : TPyTuple;
  res : PPyObject;
begin
    t := TPyTuple.Create(args);
    res := g_MyPyEngine.PyObject_Call(FHandle,t.FHandle,nil);
    g_MyPyEngine.CheckError(False);

    if res = nil then
       raise EPythonError.Create('Invoke: can''t Call!');

    Result := TPythonObject.Create(res);
end;

function TPythonObject.Invoke(args: TPyTuple): TPythonObject;
var
  res : PPyObject;
begin
    res := g_MyPyEngine.PyObject_Call(FHandle,args.FHandle,nil);
    g_MyPyEngine.CheckError(False);

    if res = nil then
       raise EPythonError.Create('Invoke: can''t Call!');

    Result := TPythonObject.Create(res);

end;

function TPythonObject.Invoke(args: TPyTuple; kw: TPyDict): TPythonObject;
var
  res : PPyObject;
begin
    if kw = nil  then  res := g_MyPyEngine.PyObject_Call(FHandle,args.FHandle,nil)
    else               res := g_MyPyEngine.PyObject_Call(FHandle,args.FHandle,kw.FHandle);
    g_MyPyEngine.CheckError(False);

    if res = nil then
       raise EPythonError.Create('Invoke: can''t Call!');

    Result := TPythonObject.Create(res);

end;

function TPythonObject.Invoke(args: TArray<TPythonObject>; kw: TPyDict): TPythonObject;
var
  t   : TPyTuple;
  res : PPyObject;
begin
    t := TPyTuple.Create(args);
    if kw = nil  then  res := g_MyPyEngine.PyObject_Call(FHandle,t.FHandle,nil)
    else               res := g_MyPyEngine.PyObject_Call(FHandle,t.FHandle,kw.FHandle);
    g_MyPyEngine.CheckError(False);

    if res = nil then
       raise EPythonError.Create('Invoke: can''t Call!');

    Result := TPythonObject.Create(res);

end;

function TPythonObject.InvokeMethod(name: string; args: TPyTuple): TPythonObject;
var
  method : TPythonObject;
begin
    method := GetAttr(name);
    try
      Result := method.Invoke(args) ;
    finally
      method.Free;
    end;
end;

function TPythonObject.InvokeMethod(name: string; args: TArray<TPythonObject> = []): TPythonObject;
var
  method : TPythonObject;
begin
    method := GetAttr(name);
    try
      Result := method.Invoke(args) ;
    finally
      method.Free;
    end;
end;

function TPythonObject.InvokeMethod(name: string; args: TArray<TPythonObject>; kw: TPyDict): TPythonObject;
var
  method : TPythonObject;
begin
    method := GetAttr(name);
    try
      Result := method.Invoke(args,kw) ;
    finally
      method.Free;
    end;
end;

function TPythonObject.InvokeMethod(name: string; args: TPyTuple; kw: TPyDict): TPythonObject;
var
  method : TPythonObject;
begin
    method := GetAttr(name);
    try
      Result := method.Invoke(args,kw) ;
    finally
      method.Free;
    end;
end;

function TPythonObject.GetItem(key: TPythonObject): TPythonObject;
var
  op : PPyObject;
begin
    op := g_MyPyEngine.PyObject_GetItem(FHandle,key.FHandle);
    if op = nil then
       raise EPythonError.Create('can''t get item!');

    Result := TPythonObject.Create(op);
end;

function TPythonObject.GetItem(key: string): TPythonObject;
var
 pyKey : TPyString;
begin
    pyKey := TPyString.Create(key);

    Result := GetItem(pyKey)
end;

function TPythonObject.GetItem(index: Integer): TPythonObject;
var
 pyKey : TPyInt;
begin
    pyKey := TPyInt.Create(index);

    Result := GetItem(pyKey)
end;


procedure TPythonObject.SetItem(key, value: TPythonObject);
var
  res : Integer;
begin
    res := g_MyPyEngine.PyObject_SetItem(FHandle,key.FHandle,value.FHandle);
    if res < 0 then
       raise EPythonError.Create('can''t set item!');
end;

procedure TPythonObject.SetItem(key: string; value: TPythonObject);
var
 pyKey : TPyString;
begin
    pyKey := TPyString.Create(key);

    SetItem(pyKey,value)
end;

procedure TPythonObject.SetItem(index: Integer; value: TPythonObject);
var
 pyKey : TPyInt;
begin
    pyKey := TPyInt.Create(index);

    SetItem(pyKey,value)
end;

function TPythonObject.AsArrayofPyObj: TArray<TPythonObject>;

    function GetItem( sequence : PPyObject; idx : Integer ) : TPythonObject;
    var
      val : PPyObject;
    begin
        val := g_MyPyEngine.PySequence_GetItem( sequence, idx );
        try
          Result := TPythonObject.Create(val);
        finally
          g_MyPyEngine.Py_XDecRef( val );
        end;
    end;

var
 vArr         : TArray<TPythonObject>;
 seq_length,i : Integer;
begin
    Result := [];
    if g_MyPyEngine.PySequence_Check( FHandle ) <> 1 then Result := [];

    seq_length := g_MyPyEngine.PySequence_Length( FHandle );
    if seq_length > 0 then
        g_MyPyEngine.Py_XDecRef( g_MyPyEngine.PySequence_GetItem( FHandle, 0 ) );

    if g_MyPyEngine.PyErr_Occurred = nil then
    begin
        //vArr := VarArrayCreate( [0, seq_length-1], varInteger );
        for i := 0 to g_MyPyEngine.PySequence_Length( FHandle )-1 do
           vArr := vArr + [ GetItem( FHandle, i ) ];
    end else
    begin
        g_MyPyEngine.PyErr_Clear;
        Result := [];
    end;

    Result := vArr;
end;

function TPythonObject.AsArrayofDouble: TArray<Double>;

    function GetSequenceItem( sequence : PPyObject; idx : Integer ) : Variant;
    var
      val : PPyObject;
    begin
        val := g_MyPyEngine.PySequence_GetItem( sequence, idx );
        try
          Result := g_MyPyEngine.PyObjectAsVariant( val );
          if (Result = Null) then
          begin
               var Ss : AnsiString := val.ob_type.tp_name;
               if string(Ss).Contains('float32') then
                 Result := g_MyPyEngine.PyFloat_AsDouble(val)

          end;

        finally
          g_MyPyEngine.Py_XDecRef( val );
        end;
    end;

var
 vArr         : Variant;
 seq_length,i : Integer;
begin
    Result := [];
    if g_MyPyEngine.PySequence_Check( FHandle ) <> 1 then Result := [];

    seq_length := g_MyPyEngine.PySequence_Length( FHandle );
    if seq_length > 0 then
        g_MyPyEngine.Py_XDecRef( g_MyPyEngine.PySequence_GetItem( FHandle, 0 ) );

    if g_MyPyEngine.PyErr_Occurred = nil then
    begin
        vArr := VarArrayCreate( [0, seq_length-1], varVariant );
        for i := 0 to g_MyPyEngine.PySequence_Length( FHandle )-1 do
           vArr[i] := GetSequenceItem( FHandle, i );
    end else // the object didn't implement the sequence API, so we return Null
    begin
        g_MyPyEngine.PyErr_Clear;
        Result := [];
    end;

    for i := 0 to  VarArrayHighBound(vArr, 1) do
       Result := Result + [ VarAsType(vArr[i],varDouble) ]

end;

function TPythonObject.AsArrayofInt: TArray<Integer>;

    function GetSequenceItem( sequence : PPyObject; idx : Integer ) : Variant;
    var
      val : PPyObject;
    begin
        val := g_MyPyEngine.PySequence_GetItem( sequence, idx );
        try
          Result := g_MyPyEngine.PyObjectAsVariant( val );
        finally
          g_MyPyEngine.Py_XDecRef( val );
        end;
    end;

var
 vArr         : Variant;
 seq_length,i : Integer;
begin
    Result := [];
    if g_MyPyEngine.PySequence_Check( FHandle ) <> 1 then Result := [];

    seq_length := g_MyPyEngine.PySequence_Length( FHandle );
    if seq_length > 0 then
        g_MyPyEngine.Py_XDecRef( g_MyPyEngine.PySequence_GetItem( FHandle, 0 ) );

    if g_MyPyEngine.PyErr_Occurred = nil then
    begin
        vArr := VarArrayCreate( [0, seq_length-1], varVariant );
        for i := 0 to g_MyPyEngine.PySequence_Length( FHandle )-1 do
           vArr[i] := GetSequenceItem( FHandle, i );
    end else // the object didn't implement the sequence API, so we return Null
    begin
        g_MyPyEngine.PyErr_Clear;
        Result := [];
    end;

    for i := 0 to  VarArrayHighBound(vArr, 1) do
      Result := Result + [ Integer (vArr[i]) ]

end;

function TPythonObject.AsArrayofString: TArray<string>;

    function GetSequenceItem( sequence : PPyObject; idx : Integer ) : Variant;
    var
      val : PPyObject;
    begin
        val := g_MyPyEngine.PySequence_GetItem( sequence, idx );
        try
          Result := g_MyPyEngine.PyObjectAsVariant( val );
        finally
          g_MyPyEngine.Py_XDecRef( val );
        end;
    end;

var
 vArr         : Variant;
 seq_length,i : Integer;
begin
    Result := [];
    if g_MyPyEngine.PySequence_Check( FHandle ) <> 1 then Result := [];

    seq_length := g_MyPyEngine.PySequence_Length( FHandle );
    if seq_length > 0 then
        g_MyPyEngine.Py_XDecRef( g_MyPyEngine.PySequence_GetItem( FHandle, 0 ) );

    if g_MyPyEngine.PyErr_Occurred = nil then
    begin
        vArr := VarArrayCreate( [0, seq_length-1], varVariant );
        for i := 0 to g_MyPyEngine.PySequence_Length( FHandle )-1 do
           vArr[i] := GetSequenceItem( FHandle, i );
    end else // the object didn't implement the sequence API, so we return Null
    begin
        g_MyPyEngine.PyErr_Clear;
        Result := [];
    end;

    for i := 0 to  VarArrayHighBound(vArr, 1) do
       Result := Result + [ vArr[i] ]

end;

function TPythonObject.AsInteger: Integer;
begin
    Result :=  g_MyPyEngine.PyObjectAsVariant(FHandle)
end;

function TPythonObject.AsString: string;
begin
    Result :=  g_MyPyEngine.PyObjectAsVariant(FHandle)
end;

function TPythonObject.AsDouble: double;
begin
    Result :=  g_MyPyEngine.PyObjectAsVariant(FHandle)
end;

function TPythonObject.AsBoolean: boolean;
begin
    Result :=  g_MyPyEngine.PyObjectAsVariant(FHandle)
end;

function TPythonObject.ToString: string;
begin
    Result := g_MyPyEngine.PyObjectAsString(FHandle);
end;

{ TPyNumber }

constructor TPyNumber.Create(_pyobject: PPyObject);
begin
    inherited Create(_pyobject);
end;

constructor TPyNumber.Create;
begin
    inherited
end;

function TPyNumber.IsNumberType(value: TPythonObject): Boolean;
begin
    Result := g_MyPyEngine.PyNumber_Check(value.FHandle) <> 0
end;

{ TPyDict }

constructor TPyDict.Create(_pyobject: PPyObject);
begin
    inherited Create(_pyobject);
end;

constructor TPyDict.Create(o: TPythonObject);
begin
    if not IsDictType(o) then
       raise EPythonError.Create('object is not a dict');

    inherited create;
    FHandle := o.FHandle;
end;

constructor TPyDict.Create;
begin
    inherited create;
    FHandle := g_MyPyEngine.PyDict_New;
    try
      if not Assigned(FHandle) then
         raise EPythonError.Create('Could not create a new dict object');
    except
      g_MyPyEngine.Py_XDECREF( FHandle );
    end;
end;

procedure TPyDict.Clear;
begin
    g_MyPyEngine.PyDict_Clear(FHandle)
end;

function TPyDict.IsDictType(value: TPythonObject): Boolean;
begin
    Result := g_MyPyEngine.PyDict_Check(value.FHandle);
end;

function TPyDict.Copy: TPyDict;
var
  op: PPyObject;
begin
    op := g_MyPyEngine.PyDict_Copy(FHandle);
    if not Assigned(op) then
         raise EPythonError.Create('Could not copy a dict object');

    Result := TPyDict.Create(op);
end;

function TPyDict.HasKey(key: TPythonObject): Boolean;
begin
    Result :=  g_MyPyEngine.PyMapping_HasKey(FHandle, key.FHandle) <> 0;
end;

function TPyDict.HasKey(key: string): Boolean;
var
  _str : TPyString;
begin
    _str := TPyString(key);
    Result := HasKey(_str)
end;

function TPyDict.Items: TPythonObject;
var
  items : PPyObject;
begin
      items := g_MyPyEngine.PyDict_Items(FHandle);
      if not Assigned(items) then
         raise EPythonError.Create('Could not get Items from object');

      Result := TPythonObject.Create(items);
end;

function TPyDict.Keys: TPythonObject;
var
  items : PPyObject;
begin
      items := g_MyPyEngine.PyDict_Keys(FHandle);
      if not Assigned(items) then
         raise EPythonError.Create('Could not get Keys from object');

      Result := TPythonObject.Create(items);
end;

function TPyDict.Values: TPythonObject;
var
  items : PPyObject;
begin
      items := g_MyPyEngine.PyDict_Values(FHandle);
      if not Assigned(items) then
         raise EPythonError.Create('Could not get Values from object');

      Result := TPythonObject.Create(items);
end;

procedure TPyDict.Update(other: TPythonObject);
var
  res : Integer;
begin
    res := g_MyPyEngine.PyDict_Update(FHandle, other.FHandle);
    if res <> 0 then
         raise EPythonError.Create('Could not update object');

end;

{ TPyString }

constructor TPyString.Create(_pyobject: PPyObject);
begin
    inherited Create(_pyobject);
end;

constructor TPyString.Create(o: TPythonObject);
begin
    if not IsStringType(o) then
       raise EPythonError.Create('object is not a string');

    inherited create;
    FHandle := o.FHandle;
end;

constructor TPyString.Create(s: string);
begin
     inherited create;
     FHandle := g_MyPyEngine.PyUnicode_FromWideString(s)
end;

function TPyString.IsStringType(value: TPythonObject): Boolean;
begin
    Result :=  g_MyPyEngine.PyString_Check(value.FHandle);
end;

{ TPyAnsiString }

constructor TPyAnsiString.Create(_pyobject: PPyObject);
begin
    inherited Create(_pyobject);
end;

constructor TPyAnsiString.Create(o: TPythonObject);
begin
    if not IsStringType(o) then
       raise EPythonError.Create('object is not a Ansistring');

    inherited create;
    FHandle := o.FHandle;
end;

constructor TPyAnsiString.Create(s: AnsiString);
begin
    inherited create;
    FHandle := g_MyPyEngine.PyString_FromStringAndSize(PAnsiChar(s), Length(s));
end;

function TPyAnsiString.IsStringType(value: TPythonObject): Boolean;
begin
    Result :=  g_MyPyEngine.PyString_Check(value.FHandle);
end;

{ TPyLong }

constructor TPyLong.Create(o: TPythonObject);
begin
    if not IsLongType(o) then
       raise EPythonError.Create('object is not a Long');

    inherited create;
    FHandle := o.FHandle;
end;

constructor TPyLong.Create(_pyobject: PPyObject);
begin
    inherited Create(_pyobject);
end;

constructor TPyLong.Create(value: Integer);
begin
    inherited create;
    FHandle := g_MyPyEngine.PyLong_FromLong(value);
end;

constructor TPyLong.Create(value: Cardinal);
begin
    inherited create;
    FHandle := g_MyPyEngine.PyLong_FromLong(value);
end;

constructor TPyLong.Create(value: Int64);
begin
    inherited create;
    FHandle := g_MyPyEngine.PyLong_FromLongLong(value);
end;

constructor TPyLong.Create(value: UInt64);
begin
    inherited create;
    FHandle := g_MyPyEngine.PyLong_FromUnsignedLongLong(value);
end;

constructor TPyLong.Create(value: Int16);
begin
    inherited create;
    FHandle := g_MyPyEngine.PyLong_FromLong(value);
end;

constructor TPyLong.Create(value: UInt16);
begin
    inherited create;
    FHandle := g_MyPyEngine.PyLong_FromLong(value);
end;

constructor TPyLong.Create(value: UInt8);
begin
    inherited create;
    FHandle := g_MyPyEngine.PyLong_FromLong(value);
end;

constructor TPyLong.Create(value: Int8);
begin
    inherited create;
    FHandle := g_MyPyEngine.PyLong_FromLong(value);
end;

constructor TPyLong.Create(value: Double);
begin
   inherited create;
   FHandle := g_MyPyEngine.PyLong_FromDouble(value);
end;

constructor TPyLong.Create(value: string);
var
  p : PAnsiChar;
begin
    inherited create;
    FHandle := g_MyPyEngine.PyLong_FromString(PAnsiChar(AnsiString(value)),p,0);
end;

function TPyLong.IsLongType(value: TPythonObject): Boolean;
begin
    Result := g_MyPyEngine.PyLong_Check(value.FHandle);
end;

function TPyLong.AsLong(value: TPythonObject): TPyLong;
var
  op : PPyObject;
begin
    op := g_MyPyEngine.PyNumber_Long(value.FHandle);
    if not Assigned(op) then
         raise EPythonError.Create('Could not cast AsLong object');

    Result := TPyLong(op);
end;

function TPyLong.ToInt16: Int16;
begin
    Result := int16(ToInt32) ;
end;

function TPyLong.ToInt32: Int32;
begin
     Result := int16(ToInt64) ;
end;

function TPyLong.ToInt64: Int64;
begin
    Result := g_MyPyEngine.PyLong_AsLongLong(FHandle);
end;

{ TPyInt }

constructor TPyInt.Create(o: TPythonObject);
begin
    if not IsIntType(o) then
       raise EPythonError.Create('object is not a Int');

    inherited create;
    FHandle := o.FHandle;
end;

constructor TPyInt.Create(_pyobject: PPyObject);
begin
    inherited Create(_pyobject);
end;

constructor TPyInt.Create(value: Cardinal);
begin
    inherited create;
    FHandle := g_MyPyEngine.PyInt_FromLong(value);
end;

constructor TPyInt.Create(value: Integer);
begin
     inherited create;
     FHandle := g_MyPyEngine.PyInt_FromLong(value);
end;

constructor TPyInt.Create(value: Int64);
begin
    inherited create;
    FHandle := g_MyPyEngine.PyLong_FromLong( int32(value) );
end;

constructor TPyInt.Create(value: UInt64);
begin
    inherited create;
    FHandle := g_MyPyEngine.PyLong_FromLong( value );
end;

constructor TPyInt.Create(value: Int16);
begin
    Create(integer(value));
end;

constructor TPyInt.Create(value: UInt16);
begin
    Create(integer(value));
end;

constructor TPyInt.Create(value: Int8);
begin
    Create(integer(value));
end;

constructor TPyInt.Create(value: UInt8);
begin
    Create(integer(value));
end;

constructor TPyInt.Create(value: string);
var
 p : PAnsiChar;
begin
    inherited create;
    FHandle := g_MyPyEngine.PyLong_FromString(PAnsiChar(AnsiString(value)),p,0);
end;

function TPyInt.AsInt(value: TPythonObject): TPyInt;
var
  op : PPyObject;
begin
    op := g_MyPyEngine.PyNumber_Int(value.FHandle);
    if not Assigned(op) then
         raise EPythonError.Create('Could not cast AsInt object');

    Result := TPyInt.Create(op);
end;

function TPyInt.IsIntType(value: TPythonObject): Boolean;
begin
    Result := g_MyPyEngine.PyInt_Check(value.FHandle);
end;

function TPyInt.ToInt16: Int16;
begin
    Result := Int16(ToInt32);
end;

function TPyInt.ToInt32: Int32;
begin
     Result := g_MyPyEngine.PyInt_AsLong(FHandle);
end;

function TPyInt.ToInt64: Int64;
begin
    Result := Int64(ToInt32);
end;

{ TPyFloat }

constructor TPyFloat.Create(_pyobject: PPyObject);
begin
    inherited Create;
    inherited Create(_pyobject);
end;

constructor TPyFloat.Create(o: TPythonObject);
begin
    if not IsFloatType(o) then
       raise EPythonError.Create('object is not a Float');

    inherited create;
    FHandle := o.FHandle;
end;

constructor TPyFloat.Create(value: Double);
begin
    inherited Create;
    FHandle := g_MyPyEngine.PyFloat_FromDouble(value);
end;

constructor TPyFloat.Create(value: string);
var
  s : TPyString;
begin
    s := TPyString.Create(value);
    FHandle := g_MyPyEngine.PyFloat_FromString(s.FHandle);
end;

function TPyFloat.IsFloatType(value: TPythonObject): Boolean;
begin
    Result := g_MyPyEngine.PyFloat_Check(value.FHandle);
end;

function TPyFloat.AsFloat(value: TPythonObject): TPyFloat;
var
  op : PPyObject;
begin
    op := g_MyPyEngine.PyNumber_Float(value.FHandle);
    if not Assigned(op) then
         raise EPythonError.Create('Could not cast AsFloat object');

    Result := TPyFloat.Create(op);
end;

{ TPySequence }

constructor TPySequence.Create;
begin
    inherited create;
end;

constructor TPySequence.Create(_pyobject: PPyObject);
begin
    inherited Create(_pyobject);
end;

function TPySequence.IsSequenceType(value: TPythonObject): Boolean;
begin
    Result := g_MyPyEngine.PySequence_Check(value.FHandle ) <> 0;
end;

function TPySequence.GetSlice(i1, i2: Integer): TPythonObject;
var
  op : PPyObject;
begin
    op := g_MyPyEngine.PySequence_GetSlice(FHandle, i1, i2);
    if not Assigned(op) then
      raise EPythonError.Create('Failed on PySequence_GetSlice');

    Result := TPythonObject.Create(op);
end;

procedure TPySequence.SetSlice(i1, i2: Integer; v: TPythonObject);
var
  r : Integer;
begin
    r := g_MyPyEngine.PySequence_SetSlice(FHandle, i1, i2, v.FHandle);
    if r < 0 then
     raise EPythonError.Create('Failed on PySequence_SetSlice');
end;

procedure TPySequence.DelSlice(i1, i2: Integer);
var
  r : Integer;
begin
    r := g_MyPyEngine.PySequence_DelSlice(FHandle, i1, i2);
    if r < 0 then
     raise EPythonError.Create('Failed on PySequence_DelSlice');
end;

function TPySequence.Index(item: TPythonObject): Integer;
var
  r : Integer;
begin
    r := g_MyPyEngine.PySequence_Index(FHandle, item.FHandle);
    if r < 0 then
    begin
        g_MyPyEngine.PyErr_Clear;
        Exit (-1)
    end;
    Result := r;
end;

function TPySequence.Contains(item: TPythonObject): Boolean;
var
  r : Integer;
begin
    r := g_MyPyEngine.PySequence_Contains(FHandle, item.FHandle);
    if r < 0 then
     raise EPythonError.Create('Failed on Contains');

   Result :=  r <> 0;
end;

function TPySequence.Concat(other: TPythonObject): TPythonObject;
var
  op : PPyObject;
begin
    op := g_MyPyEngine.PySequence_Concat(FHandle, other.FHandle);
    if not Assigned(op) then
     raise EPythonError.Create('Failed on Concat');

   Result := TPythonObject.Create(op);
end;

function TPySequence.Repeatt(count: Integer): TPythonObject;
var
  op : PPyObject;
begin
    op := g_MyPyEngine.PySequence_Repeat(FHandle, count);
    if not Assigned(op) then
     raise EPythonError.Create('Failed on Repeat');

   Result := TPythonObject.Create(op);
end;

{ TPyTuple }

constructor TPyTuple.Create;
begin
    inherited create;
    FHandle := g_MyPyEngine.PyTuple_New(0);
    if not Assigned(FHandle) then
       raise Exception.Create('TError.RaiseErrorObj: Could not create an empty tuple');
end;

constructor TPyTuple.Create(_pyobject: PPyObject);
begin
    inherited Create(_pyobject);
end;

constructor TPyTuple.Create(o: TPythonObject);
begin
    if not IsTupleType(o) then
      raise EPythonError.Create('object is not a tuple');

    inherited create;
    FHandle := o.FHandle;
end;

constructor TPyTuple.Create(const objects: array of const);
begin
    inherited create;
    FHandle := g_MyPyEngine.ArrayToPyTuple(objects);
end;

constructor TPyTuple.Create(const objects: array of PPyObject);
begin
    inherited create;
    FHandle := g_MyPyEngine.MakePyTuple(objects);
end;

constructor TPyTuple.Create(const objects: TArray<TPythonObject>);
var
  aT : array of PPyObject;
  i : Integer;
begin
    SetLength(aT, length(objects));
    for i := 0 to Length(aT) - 1 do
      aT[i] := objects[i].FHandle;

    inherited create;
    FHandle := g_MyPyEngine.MakePyTuple(aT);
end;

class function TPyTuple.IsTupleType(value: TPythonObject): Boolean;
begin
    Result := g_MyPyEngine.PyTuple_Check(value.FHandle);
end;

function TPyTuple.AsTuple(value: TPythonObject): TPyTuple;
var
  op : PPyObject;
begin
    op := g_MyPyEngine.PySequence_Tuple(value.FHandle);
    if not Assigned(op) then
         raise EPythonError.Create('Could not cast AsTuple object');

    Result := TPyTuple.Create(op);

end;

{ TPyList }

constructor TPyList.Create;
begin
    inherited create;
    FHandle := g_MyPyEngine.PyList_New(0);
    if not Assigned(FHandle) then
       raise Exception.Create('Could not create a new list object');
end;

constructor TPyList.Create(_pyobject: PPyObject);
begin
    inherited Create(_pyobject);
end;

constructor TPyList.Create(o: TPythonObject);
begin
    if not IsListType(o) then
      raise EPythonError.Create('object is not a List');

    inherited create;
    FHandle := o.FHandle;
end;

constructor TPyList.Create(const objects: array of TPythonObject);
var
  aT : array of PPyObject;
  i : Integer;
begin
    SetLength(aT, length(objects));
    for i := 0 to Length(aT) - 1 do
      aT[i] := objects[i].FHandle;

    inherited create;
    FHandle := g_MyPyEngine.MakePyList(aT);

end;

constructor TPyList.Create(const objects: array of PPyObject);
begin
     inherited create;
     FHandle := g_MyPyEngine.MakePyList(objects);
end;

function TPyList.IsListType(value: TPythonObject): Boolean;
begin
    Result := g_MyPyEngine.PyList_Check(value.FHandle);
end;

function TPyList.AsList(value: TPythonObject): TPyList;
var
  op : PPyObject;
begin
    op := g_MyPyEngine.PySequence_List(value.FHandle);
    if not Assigned(op) then
         raise EPythonError.Create('Could not cast AsList object');

    Result := TPyList.Create(op);
end;

procedure TPyList.Append(item: TPythonObject);
var
  r : Integer;
begin
    r := g_MyPyEngine.PyList_Append(FHandle, item.FHandle);
    if r < 0 then
         raise EPythonError.Create('Could not append object');
end;

procedure TPyList.Insert(index: Integer; item: TPythonObject);
var
  r : Integer;
begin
    r := g_MyPyEngine.PyList_Insert(FHandle, index, item.FHandle);
    if r < 0 then
         raise EPythonError.Create('Could not append object');
end;

procedure TPyList.Reverse;
var
  r : Integer;
begin
    r := g_MyPyEngine.PyList_Reverse(FHandle);
    if r < 0 then
     raise EPythonError.Create('Could not Reverse object');
end;

procedure TPyList.Sort;
var
  r : Integer;
begin
    r := g_MyPyEngine.PyList_Sort(FHandle);
    if r < 0 then
     raise EPythonError.Create('Could not Sort object');
end;

{ TPyIter }

constructor TPyIter.Create(_pyobject: PPyObject);
begin
    inherited Create(_pyobject);
end;

constructor TPyIter.Create(iterable: TPythonObject);
begin
    inherited create;
    FHandle := g_MyPyEngine.PyObject_GetIter(iterable.FHandle);
    if not Assigned(FHandle) then
         raise EPythonError.Create('object is not a Iter');
end;

function TPyIter.GetCurrent: TObject;
begin
    Result := FCurrent;
end;

function TPyIter.MoveNext: Boolean;
var
  next : PPyObject;
begin
    // dispose of the previous object, if there was one
    if FCurrent <> nil then
    begin
        FCurrent.Free;
        FCurrent := nil;
    end;

    next := g_MyPyEngine.PyIter_Next(FHandle);
    if next = nil  then
    begin
        Result := False;
        Exit;
    end;

    FCurrent := TPythonObject.Create(next);
    Result := True;
end;

function TPyIter.QueryInterface(const IID: TGUID; out Obj): HResult;
begin
    if GetInterface(IID, Obj) then
    Result:= S_OK
  else
    Result:= E_NOINTERFACE;
end;

procedure TPyIter.Reset;
begin

end;

function TPyIter._AddRef: Integer;
begin
    Result := -1
end;

function TPyIter._Release: Integer;
begin
    Result := -1
end;

initialization

finalization
  if g_MyPyEngine <> nil then
     g_MyPyEngine.Free;


end.
