{*******************************************************}
{                                                       }
{       Keras Wrapper for Delphi                        }
{                                                       }
{       Copyright (C) 2020 Pigrecos(Max)                }
{                                                       }
{*******************************************************}
unit Keras;

interface
  uses System.Generics.Collections,System.Rtti, System.SysUtils, System.TypInfo, Winapi.Windows,
       PythonEngine,
       Python.Utils,
       Models ;

type

  TKeras = class
   protected
      class function ToPython(value: TValue): TPythonObject; static;
      class function ToDict(input:  TDictionary<string,string>): TPyDict;  overload; static;
      class function ToDict(input:  TDictionary<Integer,Double>): TPyDict; overload; static;
      class function ToList(input: TArray<TValue>): TPyList; static;
      class function ToTuple(input: TArray<TValue>): TPyTuple; static;
    public
       class var hKerasMod     : TPythonObject;
       class var hTensorFlowMod: TPythonObject;
       class var hkeras2onnxMod: TPythonObject;
       class var htfjsMod      : TPythonObject;

       constructor Create;
  end;

  TBase = class(TKeras)
    private
      function  Instantiate: TPythonObject;
      function  GetItem(name: string): TValue;
      procedure SetItem(name: string; const Value: TValue);

    public
       PyInstance : TPythonObject;
       Parameters : TList< TPair<AnsiString,TValue> >;

       constructor Create;
       destructor  Destroy; override;

       function  InvokeMethod(                                   method: string; args: TList< TPair<AnsiString,TValue> >): TPythonObject;
       class function  InvokeStaticMethod(caller: TPythonObject; method: string; args: TList< TPair<AnsiString,TValue> >): TPythonObject;
       procedure Init;

       class function  GetKerasClassIstance(nameClass: AnsiString): TPythonObject;
       class function  GetTFClassIstance(nameClass: AnsiString): TPythonObject;
       class function  GetTFJSClassIstance(nameClass: AnsiString): TPythonObject; static;
       class function  GetOnnxClassIstance(nameClass: AnsiString):TPythonObject;

       property Item[name :string]:TValue read GetItem write SetItem; default;

  end;

  TKerasIterator = class;

  //UtilsClass

  TCustomObjectScope = class(TBase)
     public
       constructor Create; overload;
       constructor Create(py: PPyObject); overload;
       constructor Create(py: TPythonObject); overload;
  end;

  THDF5Matrix = class(TBase)
     public
       constructor Create;overload;
       constructor Create(py: PPyObject); overload;
       constructor Create(py: TPythonObject); overload;
       constructor Create(datapath: string;dataset: string;start : Integer = 0;fine: PInteger= nil; normalizer: Pointer= nil); overload;

  end;

  TSequence = class(TBase)
     public
       constructor Create;overload;
       constructor Create(py: PPyObject); overload;
       constructor Create(py: TPythonObject); overload;
       constructor Create(py: TKerasIterator); overload;
       constructor Create(py: TNDArray); overload;
  end;

  // InternalTypes

  TStringOrInstance = class
     public
       PyObject : TPythonObject;
       constructor Create(py: PPyObject); overload;
       constructor Create(py: TPythonObject); overload;
       constructor Create(opt: string); overload;
       constructor Create(opt: TBase); overload;
  end;

  TKerasFunction = class
     public
       PyObject : TPythonObject;
       constructor Create(py: PPyObject); overload;
       constructor Create(py: TPythonObject); overload;
  end;

  TKerasIterator = class(TPythonObject)
     public
       PyObject : TPythonObject;
       constructor Create(py: PPyObject); overload;
       constructor Create(py: TPythonObject); overload;
  end;

  TDirectoryIterator = class(TPythonObject)
     public
       PyObject : TPythonObject;
       constructor Create(py: PPyObject); overload;
       constructor Create(py: TPythonObject); overload;
  end;

  // Keras.Initializer

  TZeros = class(TBase)
     public
       constructor Create;
  end;

  TOnes = class(TBase)
     public
       constructor Create;
  end;

  TConstant = class(TBase)
     public
       constructor Create(value : Double = 0.0);
  end;

  TRandomNormal = class(TBase)
     public
       constructor Create(mean : Double = 0.0; stddev: Double= 0.05; seed : PInteger= nil);
  end;

  TRandomUniform = class(TBase)
     public
       constructor Create(minval : Double = 0.0; maxval: Double= 0.05; seed : PInteger= nil);
  end;

  TTruncatedNormal = class(TBase)
     public
       constructor Create(mean : Double = 0.0; stddev: Double= 0.05; seed : PInteger= nil);
  end;

  TVarianceScaling = class(TBase)
     public
       constructor Create(scale: Double = 1.0; mode: string = 'fan_in'; distribution: string = 'normal'; seed : PInteger= nil);
  end;

  TOrthogonal = class(TBase)
     public
       constructor Create(gain: Double = 1.0; seed : PInteger= nil);
  end;

  TIdentity = class(TBase)
     public
       constructor Create(gain: Double = 1.0);
  end;

  TLecunUniform = class(TBase)
     public
       constructor Create(seed: PInteger = nil);
  end;

  TGlorotNormal = class(TBase)
     public
       constructor Create(seed: PInteger = nil);
  end;

  TGlorotUniform = class(TBase)
     public
       constructor Create(seed: PInteger = nil);
  end;

  THeUniform = class(TBase)
     public
       constructor Create(seed: PInteger = nil);
  end;

  THeNormal = class(TBase)
     public
       constructor Create(seed: PInteger = nil);
  end;

  TLecunNormal = class(TBase)
     public
       constructor Create(seed: PInteger = nil);
  end;

  // Regularizer

  TL1L2 = class(TBase)
     public
       constructor Create(l1 : Double = 0.01; l2 : Double = 0.01);
  end;

  TL1 = class(TBase)
     public
       constructor Create(l1 : Double = 0.01);
  end;

  TL2 = class(TBase)
     public
       constructor Create(l2 : Double = 0.01);
  end;

  // Optimizers

  TSGD = class(TBase)
     public
       constructor Create(lr: Double= 0.01; momentum: Double = 0.0; decay : Double = 0.0; nesterov : Boolean = false);
  end;

  TRMSprop = class(TBase)
     public
       constructor Create(lr: Double= 0.01; rho: Double = 0.9; epsilon : PDouble = nil; decay : Double = 0.0);
  end;

  TAdagrad = class(TBase)
     public
       constructor Create(lr: Double= 0.01; epsilon : PDouble = nil; decay : Double = 0.0);
  end;

  TAdadelta = class(TBase)
     public
       constructor Create(lr: Double= 1.0; rho: Double = 0.95; epsilon : PDouble = nil; decay : Double = 0.0);
  end;

  TAdam = class(TBase)
     public
       constructor Create(lr:Double = 0.001; beta_1: Double= 0.9; beta_2:Double= 0.999; epsilon: PDouble = nil; decay: double=0.0; amsgrad: Boolean = false);
  end;

  TAdamax = class(TBase)
     public
       constructor Create(lr:Double = 0.002; beta_1: Double= 0.9; beta_2:Double= 0.999; epsilon: PDouble = nil; decay: double=0.0);
  end;

  TNadam = class(TBase)
     public
       constructor Create(lr:Double = 0.002; beta_1: Double= 0.9; beta_2:Double= 0.999; epsilon: PDouble = nil; schedule_decay: double=0.004);
  end;

  // Metrics

  TMetrics = class(TBase)
     public
       caller : TPythonObject;
       constructor Create;

       function  MSE (y_true: TNDarray; y_pred: TNDarray) :  TNDarray;
       function  MAE (y_true: TNDarray; y_pred: TNDarray) :  TNDarray;
       function  MAPE(y_true: TNDarray; y_pred: TNDarray) :  TNDarray;
       function  MSLE(y_true: TNDarray; y_pred: TNDarray) :  TNDarray;
       function  Cosine(y_true: TNDarray; y_pred: TNDarray) :  TNDarray;
       function  BinaryAccuracy(y_true: TNDarray; y_pred: TNDarray) :  TNDarray;
       function  CategoricalAccuracy(y_true: TNDarray; y_pred: TNDarray) :  TNDarray;
       function  SparseCategoricalAccuracy(y_true: TNDarray; y_pred: TNDarray) :  TNDarray;
       function  TopKCategoricalAccuracy(y_true: TNDarray; y_pred: TNDarray; k: Integer = 5) :  TNDarray;
       function  SparseTopKCategoricalAccuracy(y_true: TNDarray; y_pred: TNDarray; k: Integer = 5) :  TNDarray;
  end;

  // Losses

  TLosses = class(TBase)
     public
       caller : TPythonObject;
       constructor Create;

       function MeanSquaredError(y_true:TNDarray ; y_pred:TNDarray ):TNDarray ;
       function MeanAbsoluteError(y_true:TNDarray ; y_pred:TNDarray ):TNDarray ;
       function MeanAbsolutePercentageError(y_true:TNDarray ; y_pred:TNDarray ):TNDarray;
       function MeanSquaredLogarithmicError(y_true:TNDarray ; y_pred:TNDarray ):TNDarray;
       function SquaredHinge(y_true:TNDarray ; y_pred:TNDarray ):TNDarray;
       function Hinge(y_true:TNDarray ; y_pred:TNDarray ):TNDarray;
       function CategoricalHinge(y_true:TNDarray ; y_pred:TNDarray ):TNDarray;
       function LogCosh(y_true:TNDarray ; y_pred:TNDarray ):TNDarray;
       function CategoricalCrossentropy(y_true:TNDarray ; y_pred:TNDarray ):TNDarray;
       function SparseCategoricalCrossentropy(y_true:TNDarray ; y_pred:TNDarray ):TNDarray ;
       function BinaryCrossentropy(y_true:TNDarray ; y_pred:TNDarray ):TNDarray;
       function KullbackLeiblerDivergence(y_true:TNDarray ; y_pred:TNDarray ):TNDarray ;
       function Poisson(y_true:TNDarray ; y_pred:TNDarray ):TNDarray;
       function CosineProximity(y_true:TNDarray ; y_pred:TNDarray ):TNDarray;
  end;

  //Datasets

  TBostonHousing = class(TBase)
     public
       caller : TPythonObject;
       constructor Create;
       function load_data(path: string = 'boston_housing.npz'; test_split: double = 0.2; seed : Integer = 113): TArray<TNDArray>;
  end;

  TCifar10 = class(TBase)
     public
       caller : TPythonObject;
       constructor Create;
       class function load_data: TArray<TNDArray>;
  end;

  TCifar100 = class(TBase)
     public
       caller : TPythonObject;
       constructor Create;
       class function load_data(label_mode: string = 'fine'): TArray<TNDArray>;
  end;

  TFashionMNIST = class(TBase)
     public
       caller : TPythonObject;
       constructor Create;
       class function load_data: TArray<TNDArray>;
  end;

  TMNIST = class(TBase)
     public
       caller : TPythonObject;
       constructor Create;
       class function load_data(path: string = 'mnist.npz'): TArray<TNDArray>;
  end;

  TIMDB = class(TBase)
     public
       caller : TPythonObject;
       constructor Create;
       function GetWordIndex(path: string= 'imdb_word_index.json'): TDictionary<string, Integer>;
       function load_data(path      : string= 'imdb.npz';
                          num_words : PInteger= nil;
                          skip_top  : Integer= 0;
                          maxlen    : PInteger= nil;
                          seed      : Integer= 113;
                          start_char: Integer= 1;
                          oov_char  : Integer= 2;
                          index_from: Integer= 3): TArray<TNDArray>;
  end;

  TReuters = class(TBase)
     public
       caller : TPythonObject;
       constructor Create;
       function load_data(path      : string= 'reuters.npz';
                          num_words : PInteger= nil;
                          skip_top  : Integer= 0;
                          maxlen    : PInteger= nil;
                          test_split: Double = 0.2;
                          seed      : Integer= 113;
                          start_char: Integer= 1;
                          oov_char  : Integer= 2;
                          index_from: Integer= 3): TArray<TNDArray>;
  end;

  // Constraints

  TMaxNorm = class(TBase)
     public
       constructor Create(max_value: double= 2; axis: Integer= 0);
  end;

  TNonNeg = class(TBase)
     public
       constructor Create;
  end;

  TUnitNorm = class(TBase)
     public
       constructor Create(axis: Integer= 0);
  end;

  TMinMaxNorm = class(TBase)
     public
       constructor Create(min_value:Double= 0.0; max_value: Double= 1.0; rate: Double= 1.0; axis: Integer = 0);
  end;

  //Callbacks

  TCallback = class(TBase)
     public
       constructor Create;overload;
       constructor Create(py: PPyObject); overload;
       constructor Create(py: TPythonObject); overload;
       class function Custom(name: string;fileOrcode: string; isFile: Boolean = true): TCallback;
       function  GetDoubleArray(prop : string): TArray<Double>;
  end;

  TBaseLogger = class(TCallback)
     public
       constructor Create(stateful_metrics: TArray<string>);
  end;

  TTerminateOnNaN = class(TCallback)
     public
       constructor Create;
  end;

  TProgbarLogger = class(TCallback)
     public
       constructor Create(count_mode:string = 'samples'; stateful_metrics: TArray<string> = []);
  end;

  THistory = class(TCallback)
  private
    function GetEpoch: TArray<Integer>;
    function GetHistoryLogs: TDictionary<string, TArray<Double>>;
     public
       FEpoch       : TArray<Integer>;
       FHistoryLogs : TDictionary< string,TArray<Double> >;
       constructor Create;overload;
       constructor Create(py: PPyObject); overload;
       constructor Create(py: TPythonObject); overload;

       property Epoch      : TArray<Integer>                      read GetEpoch;
       property HistoryLogs: TDictionary< string,TArray<Double> > read GetHistoryLogs;
  end;

  TModelCheckpoint = class(TCallback)
     public
       constructor Create(filepath         : string;
                          monitor          : string = 'val_loss';
                          verbose          : Integer = 0;
                          save_best_only   : Boolean = true;
                          save_weights_only: Boolean = false;
                          mode             : string = 'auto';
                          period           : Integer  = 1);
  end;

  TEarlyStopping = class(TCallback)
     public
       constructor Create(monitor  : string = 'val_loss';
                          min_delta: Double = 0;
                          patience : Integer= 0;
                          verbose  : Integer= 0;
                          mode     : string = 'auto';
                          baseline : PDouble= nil;
                          restore_best_weights: Boolean = false);
  end;

  TRemoteMonitor = class(TCallback)
     public
       constructor Create(root   : string = 'http://localhost:9000';
                          path   : string = '/publish/epoch/end/';
                          field  : string = 'data';
                          headers: TDictionary<string, string> = nil;
                          send_as_json : Boolean = false);
  end;

  TFunSchedule = function (EpochIdx: Integer): Double;

  TLearningRateScheduler = class(TCallback)
     public
       constructor Create(schedule:TFunSchedule; verbose: Integer= 0);
  end;

  TTensorBoard = class(TCallback)
     public
       constructor Create(log_dir               : string  = './logs';
                          histogram_freq        : Integer = 0;
                          batch_size            : Integer = 32;
                          write_graph           : Boolean = true;
                          write_grads           : Boolean= false;
                          write_images          : Boolean= false;
                          embeddings_freq       : Integer= 0;
                          embeddings_layer_names: TArray<String> = nil;
                          embeddings_metadata   : TDictionary<string, string> = nil;
                          embeddings_data       : TNDarray = nil;
                          update_freq           : string = 'epoch');
  end;

  TReduceLROnPlateau = class(TCallback)
     public
       constructor Create(monitor  : string  = 'val_loss';
                          factor   : Double  = 0.1;
                          patience : Integer = 10;
                          verbose  : Integer = 0;
                          mode     : string  = 'auto';
                          min_delta: Double  = 0.0001;
                          cooldown : Integer = 0;
                          min_lr   : Double  = 0);
  end;

  TCSVLogger = class(TCallback)
     public
       constructor Create(filename: string; separator:string = ','; append : Boolean= false);
  end;

  TBackend = class(TBase)
     public
       caller : TPythonObject;
       constructor Create;
       function  GetBackend: string;
       function  Epsilon: Double;
       procedure SetEpsilon(e: Double);
       function  FloatX: string;
       procedure SetFloatX(floatx: string) ;
       function  CastToFloatX(x: TNDarray): TNDarray ;
       function  ImageDataFormat: string;
       procedure DisableEager;
       procedure SetImageDataFormat(data_format: string);
       function  GetUid(prefix : string = ''): string;
       procedure ResetUids;
       procedure ClearSession;
       function  IsSparse(tensor: TNDarray): Boolean;
       function  ToDense(tensor: TNDarray): TNDarray;
       function  Cast(x: TPythonObject; dtype: string = 'float32'): TPythonObject;
  end;

  function ImportModule(Name: string): PPyObject;

implementation
    uses System.IOUtils, np.Api,np.Base, utils ;

function ImportModule(Name: string): PPyObject;
var
  res : PPyObject;
begin
    res := g_MyPyEngine.PyImport_ImportModule(PAnsiChar(AnsiString(Name)) ) ;

    if g_MyPyEngine.PyErr_Occurred <> nil then
    begin
        g_MyPyEngine.PyErr_Clear;
        Exit(nil);
    end;
    Result := res;
end;

{ TKeras }

constructor TKeras.Create;
begin
    if Assigned(hKerasMod) then Exit;

    hKerasMod :=  TPythonObject.Create( ImportModule('keras') );

    try
      hTensorFlowMod:=  TPythonObject.Create( ImportModule('tensorflow') );
    except
      MessageBoxA(0,'Warning! tensorflow is not installed. Required to load models','Warning',MB_OK)
    end;

    try
      hkeras2onnxMod:=  TPythonObject.Create( ImportModule('onnxmltools') );
    except
       MessageBoxA(0,'Warning! onnxmltools is not installed','Warning',MB_OK)
    end;

    try
      htfjsMod      :=  TPythonObject.Create( ImportModule('tensorflowjs') );
    except
       MessageBoxA(0,'Warning! tensorflowjs is not installed','Warning',MB_OK)
    end;
end;

class function TKeras.ToPython(value: TValue): TPythonObject;
var
  Info     : PTypeInfo;
  data     : PTypeData;
  aTmp     : TArray<TValue>;
begin
    Result := nil;

    Info := value.TypeInfo;
    data := value.TypeData;

    //PPyObject type
    if Info.Kind = tkPointer then
    begin
        var pp : PPyObject;
        if info.Name = 'PPyObject' then
        begin
          pp := value.AsType<PPyObject> ;
          Result := TPythonObject.Create(pp);
        end;

    end
    //TPythonObject type
    else if Info.Kind = tkClass then
    begin
        if Info^.Name = 'TPythonObject' then  Exit(value.AsType<TPythonObject>);

        if (data.ParentInfo^.Name = 'TPythonObject') or (data.ParentInfo^.Name = 'TNDArray') or (data.ParentInfo^.Name = 'TPySequence')then
           Result := value.AsType<TPythonObject>
        else if (data.ParentInfo^.Name = 'TBase') or (data.ParentInfo^.Name = 'TBaseLayer')  then
           Result := value.AsType<TBase>.PyInstance
        // TList<System.string>
        else if Info.Name = 'TList<System.string>' then
        begin
            var aTmpStr    : TArray<string>;
            var i          : Integer;

            aTmpStr := TList<string>(value.AsObject).ToArray;
            for i := 0 to High(aTmpStr) do
              aTmp := aTmp + [ aTmpStr[i] ];

            Result := TPythonObject.Create( g_MyPyEngine.ArrayToPyList( TValueArrayToArrayOfConst( aTmp ) ) );
        end
        else if info.Name = 'TBaseLayer'  then
        begin
            Result := value.AsType<TBase>.PyInstance
        end
        else if info.Name = 'TKInput'  then
        begin
            Result := value.AsType<TBase>.PyInstance
        end
        else if info.Name = 'TBase'  then
        begin
            Result := value.AsType<TBase>.PyInstance
        end
        else if info.Name = 'TSequence'  then
        begin
            Result := value.AsType<TSequence>.PyInstance
        end
        else if info.Name = 'TStringOrInstance'  then
        begin
            Result := value.AsType<TStringOrInstance>.PyObject
        end
        else if info.Name = 'TKerasFunction'  then
        begin
            Result := value.AsType<TKerasFunction>.PyObject
        end;

    end
    // array type
    else if (Info.Kind = tkDynArray)  or (Info.Kind = tkArray) then
    begin
        Result := ToList( value.AsArray );
    end
    // tnp_slice type
    else if Info = System.TypeInfo(Tnp_Slice)  then
    begin
       Result := value.AsType<Tnp_Slice>.ToPython ;
    end
    // tnp_shape type
    else if Info = System.TypeInfo(Tnp_Shape)  then
    begin
        Result := ToTuple ( TValue.ArrayOfToValueArray<Integer>( value.AsType<Tnp_Shape>.Dimensions) );
    end
    // double type
    else if (Info = System.TypeInfo(Double)) or (Info^.Kind = tkFloat)         then
    begin
        Result    := TPyFloat.Create(value.AsType<Double>);
    end
    // int64 type
    else if Info = System.TypeInfo(Int64)         then
    begin
        Result    := TPyLong.Create(value.AsInt64);
    end
    // integer type
    else if Info = System.TypeInfo(integer) then
    begin
        Result    := TPyInt.Create(value.AsInteger);
    end
    else if Info = System.TypeInfo(Byte)         then
    begin
        Result    := TPyInt.Create(value.AsInteger);
    end
    else if Info = System.TypeInfo(Word)         then
    begin
        Result    := TPyInt.Create(value.AsInteger);
    end
    // string type
    else if (Info = System.TypeInfo(string)) or (Info^.Name = 'string') then
    begin
        Result   := TPyString.Create(value.AsString);
    end
    //ansistring type
    else if Info = System.TypeInfo(AnsiString)         then
    begin
        Result    := TPyString.Create(value.AsString);
    end
    else if Info = System.TypeInfo(Boolean) then
    begin
      if value.AsBoolean = true then
        Result := TPythonObject.Create(PPyObject(g_MyPyEngine.Py_True) )
      else
         Result := TPythonObject.Create(PPyObject(g_MyPyEngine.Py_False) );
      g_MyPyEngine.Py_XIncRef(Result.Handle);
    end;

    if Result = nil then
      raise Exception.Create('Error in ToPython conversion function');
end;

class function TKeras.ToTuple(input: TArray<TValue>): TPyTuple;
var
 aArray : TArray<TPythonObject>;
 i : Integer;
begin
    SetLength(aArray,Length(input)) ;
    for i := 0 to  Length(input) - 1 do
     aArray[i] := ToPython(input[i]);

    Result := TPyTuple.Create(aArray);
end;

class function TKeras.ToList(input: TArray<TValue>): TPyList;
var
 aArray : TArray<TPythonObject>;
 i : Integer;
begin
    SetLength(aArray,Length(input)) ;
    for i := 0 to  Length(input) - 1 do
     aArray[i] := ToPython(input[i]);

    Result := TPyList.Create(aArray);
end;

class function TKeras.ToDict(input:  TDictionary<Integer,Double>): TPyDict;
var
 dict : TPyDict;
 item : TPair<Integer,Double>;

begin
    dict := TPyDict.Create;

    for item in input do
      dict[ToPython(item.Key)] := ToPython(item.Value);

    Result := dict;
end;

class function TKeras.ToDict(input:  TDictionary<string,string>): TPyDict;
var
 dict : TPyDict;
 item : TPair<string,string>;

begin
    dict := TPyDict.Create;

    for item in input do
      dict[ToPython(item.Key)] := ToPython(item.Value);

    Result := dict;
end;

{ TBase }

constructor TBase.Create;
begin
    inherited Create;

    Parameters := TList< TPair<AnsiString,TValue> >.Create;

end;

destructor TBase.Destroy;
begin
    Parameters.Free;

    Free;
end;

class function TBase.GetKerasClassIstance(nameClass: AnsiString):TPythonObject;
var
  cClass : PPyObject;
  spilt  : TArray<String>;
  i      : Integer;
begin
    spilt := string(nameClass).split(['.']) ;
    cClass := hKerasMod.Handle;

    for i := 0 to High(spilt) do
       cClass := g_MyPyEngine.PyObject_GetAttrString(cClass, PAnsiChar( AnsiString(spilt[i]) )) ;

    if cClass = nil then Result := nil
    else                 Result := TPythonObject.Create(cClass)
end;

class function TBase.GetTFClassIstance(nameClass: AnsiString):TPythonObject;
var
  cClass : PPyObject;
  spilt  : TArray<String>;
  i      : Integer;
begin
    spilt := string(nameClass).split(['.']) ;
    cClass := hTensorFlowMod.Handle;

    for i := 0 to High(spilt) do
       cClass := g_MyPyEngine.PyObject_GetAttrString(cClass, PAnsiChar( AnsiString(spilt[i]) )) ;

    if cClass = nil then Result := nil
    else                 Result := TPythonObject.Create(cClass)
end;

class function TBase.GetTFJSClassIstance(nameClass: AnsiString):TPythonObject;
var
  cClass : PPyObject;
  spilt  : TArray<String>;
  i      : Integer;
begin
    spilt := string(nameClass).split(['.']) ;
    cClass := htfjsMod.Handle;

    if cClass = nil then Exit(nil);

    for i := 0 to High(spilt) do
       cClass := g_MyPyEngine.PyObject_GetAttrString(cClass, PAnsiChar( AnsiString(spilt[i]) )) ;

    if cClass = nil then Result := nil
    else                 Result := TPythonObject.Create(cClass)
end;

class function TBase.GetOnnxClassIstance(nameClass: AnsiString):TPythonObject;
var
  cClass : PPyObject;
  spilt  : TArray<String>;
  i      : Integer;
begin
    spilt := string(nameClass).split(['.']) ;
    cClass := hkeras2onnxMod.Handle;

    for i := 0 to High(spilt) do
       cClass := g_MyPyEngine.PyObject_GetAttrString(cClass, PAnsiChar( AnsiString(spilt[i]) )) ;

    if cClass = nil then Result := nil
    else                 Result := TPythonObject.Create(cClass)
end;

function TBase.GetItem(name: string): TValue;
var
  i : Integer;
begin
    for i := 0 to Parameters.Count-1 do
    begin
      if Parameters[i].Key = name then
        Exit(Parameters[i].Value);
    end;

    raise Exception.Create('Parameters not Contains key: '+ name);

end;

procedure TBase.SetItem(name: string; const Value: TValue);
begin
    Parameters.Add( TPair<AnsiString,TValue>.Create(name,value) );
end;

procedure TBase.Init;
begin
    PyInstance := Instantiate;
end;

function TBase.Instantiate: TPythonObject;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   skip   : Boolean;
   item   : TPair<AnsiString,TValue>;
begin

  if Parameters.Count > 0 then pyargs := ToTuple([Parameters.First.Value])
  else                         pyargs := ToTuple([]);

  kwargs := TPyDict.Create;

  skip := True;
  for item in  Parameters do
  begin
      if skip then
      begin
          skip := False ;
          Continue;
      end;

      if (item.Value.IsEmpty) or ( (item.Value.IsType<string>) and (string.IsNullOrWhiteSpace(item.Value.AsString)) ) then Continue;

      if ( (item.Value.IsType<TPythonObject>) and (item.Value.AsType<TPythonObject>.IsNone))  then Continue;

      kwargs[item.Key]:= ToPython(item.Value);
  end;

  if Parameters.Count > 0 then Result := PyInstance.Invoke(pyargs,kwargs)
  else                         Result := PyInstance.Invoke(pyargs,nil)

end;

class function TBase.InvokeStaticMethod(caller : TPythonObject; method: string; args: TList< TPair<AnsiString,TValue> >): TPythonObject;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   skip   : Boolean;
   item   : TPair<AnsiString,TValue>;
begin

  if args.Count > 0 then pyargs := ToTuple([args.First.Value])
  else                   pyargs := ToTuple([]);

  kwargs := TPyDict.Create;

  skip := True;
  for item in  args do
  begin
      if skip then
      begin
          skip := False ;
          Continue;
      end;

      if (item.Value.IsEmpty) or ( (item.Value.IsType<string>) and (string.IsNullOrWhiteSpace(item.Value.AsString)) ) then Continue;

      if ( (item.Value.IsType<TPythonObject>) and (item.Value.AsType<TPythonObject>.IsNone))  then Continue;

      kwargs[item.Key]:= ToPython(item.Value);
  end;

  if args.Count > 0 then Result := caller.InvokeMethod(method,pyargs,kwargs)
  else                   Result := caller.InvokeMethod(method,pyargs,nil)

end;

function TBase.InvokeMethod(method: string; args: TList< TPair<AnsiString,TValue> >): TPythonObject;
var
   pyargs : TPyTuple;
   kwargs : TPyDict;
   skip   : Boolean;
   item   : TPair<AnsiString,TValue>;
begin

  if args.Count > 0 then pyargs := ToTuple([args.First.Value])
  else                   pyargs := ToTuple([]);

  kwargs := TPyDict.Create;

  skip := True;
  for item in  args do
  begin
      if skip then
      begin
          skip := False ;
          Continue;
      end;

      if (item.Value.IsEmpty) or ( (item.Value.IsType<string>) and (string.IsNullOrWhiteSpace(item.Value.AsString)) ) then Continue;

      if ( (item.Value.IsType<TPythonObject>) and (item.Value.AsType<TPythonObject>.IsNone))  then Continue;

      kwargs[item.Key]:= ToPython(item.Value);
  end;

  if args.Count > 0 then Result := PyInstance.InvokeMethod(method,pyargs,kwargs)
  else                   Result := PyInstance.InvokeMethod(method,pyargs,nil) ;

end;

{ TCustomObjectScope }

constructor TCustomObjectScope.Create;
begin
    inherited Create;
    PyInstance := GetKerasClassIstance('utils.CustomObjectScope');
end;

constructor TCustomObjectScope.Create(py: PPyObject);
begin
    inherited Create;
    PyInstance := TPythonObject.Create( py );
end;

constructor TCustomObjectScope.Create(py: TPythonObject);
begin
    inherited Create;
    PyInstance := py;
end;

{ THDF5Matrix }

constructor THDF5Matrix.Create;
begin
    inherited Create;
    PyInstance := GetKerasClassIstance('utils.HDF5Matrix');
end;

constructor THDF5Matrix.Create(py: PPyObject);
begin
    inherited Create;
    PyInstance := TPythonObject.Create( py );
end;

constructor THDF5Matrix.Create(py: TPythonObject);
begin
    inherited Create;
    PyInstance := py;
end;

constructor THDF5Matrix.Create(datapath, dataset: string; start: Integer; fine: PInteger; normalizer: Pointer);
begin
    inherited Create;
    PyInstance := GetKerasClassIstance('utils.HDF5Matrix');
    { TODO -oMax -c :  costruttore incompleto 23/02/2020 19:51:03 }
end;

{ TSequence }

constructor TSequence.Create;
begin
    inherited Create;
    PyInstance := GetKerasClassIstance('utils.Sequence');
end;

constructor TSequence.Create(py: PPyObject);
begin
    inherited Create;
    PyInstance := TPythonObject.Create( py );
end;

constructor TSequence.Create(py: TPythonObject);
begin
    inherited Create;
    PyInstance := py;
end;

constructor TSequence.Create(py: TKerasIterator);
begin
    Create;
    PyInstance :=  py.PyObject;
end;

constructor TSequence.Create(py: TNDArray);
begin
    Create(py.Handle)
end;

{ TStringOrInstance }

constructor TStringOrInstance.Create(py: PPyObject);
begin
    PyObject := TPythonObject.Create( py );
end;

constructor TStringOrInstance.Create(py: TPythonObject);
begin
    PyObject := py;
end;

constructor TStringOrInstance.Create(opt: TBase);
begin
    Create(opt.PyInstance);
end;

constructor TStringOrInstance.Create(opt: string);
begin
    Create( TKeras.ToPython(opt) );
end;

{ TKerasFunction }

constructor TKerasFunction.Create(py: TPythonObject);
begin
    PyObject := py;
end;

constructor TKerasFunction.Create(py: PPyObject);
begin
    PyObject := TPythonObject.Create( py );
end;

{ TKerasIterator }

constructor TKerasIterator.Create(py: TPythonObject);
begin
    PyObject := py;
end;

constructor TKerasIterator.Create(py: PPyObject);
begin
    PyObject := TPythonObject.Create( py );
end;

{ TDirectoryIterator }

constructor TDirectoryIterator.Create(py: TPythonObject);
begin
    PyObject := py;
end;

constructor TDirectoryIterator.Create(py: PPyObject);
begin
    PyObject := TPythonObject.Create( py );
end;

{ TZeros }

constructor TZeros.Create;
begin
    inherited Create;
    PyInstance := GetKerasClassIstance('initializers.Zeros');
    init();
end;

{ TOnes }

constructor TOnes.Create;
begin
    inherited Create;
    PyInstance := GetKerasClassIstance('initializers.Ones');
    init();
end;

{ TConstant }

constructor TConstant.Create(value : Double = 0.0);
begin
    inherited Create;

    Parameters.Add( TPair<AnsiString,TValue>.Create('value',value) );

    PyInstance := GetKerasClassIstance('initializers.Constant');
    init();
end;

{ TRandomNormal }

constructor TRandomNormal.Create(mean, stddev: Double; seed: PInteger);
begin
    inherited Create;

    Parameters.Add( TPair<AnsiString,TValue>.Create('mean',mean) );
    Parameters.Add( TPair<AnsiString,TValue>.Create('stddev',stddev) );

    if seed <> nil then
        Parameters.Add( TPair<AnsiString,TValue>.Create('seed',seed^) );

    PyInstance := GetKerasClassIstance('initializers.RandomNormal');
    init();
end;

{ TRandomUniform }

constructor TRandomUniform.Create(minval, maxval: Double; seed: PInteger);
begin
    inherited Create;

    Parameters.Add( TPair<AnsiString,TValue>.Create('minval',minval) );
    Parameters.Add( TPair<AnsiString,TValue>.Create('maxval',maxval) );

    if seed <> nil then
        Parameters.Add( TPair<AnsiString,TValue>.Create('seed',seed^) );

    PyInstance := GetKerasClassIstance('initializers.RandomUniform');
    init();
end;

{ TTruncatedNormal }

constructor TTruncatedNormal.Create(mean, stddev: Double; seed: PInteger);
begin
    inherited Create;

    Parameters.Add( TPair<AnsiString,TValue>.Create('mean',mean) );
    Parameters.Add( TPair<AnsiString,TValue>.Create('stddev',stddev) );

    if seed <> nil then
        Parameters.Add( TPair<AnsiString,TValue>.Create('seed',seed^) );

    PyInstance := GetKerasClassIstance('initializers.TruncatedNormal');
    init();
end;

{ TVarianceScaling }

constructor TVarianceScaling.Create(scale: Double; mode, distribution: string; seed: PInteger);
begin
    inherited Create;

    Parameters.Add( TPair<AnsiString,TValue>.Create('scale',scale) );
    Parameters.Add( TPair<AnsiString,TValue>.Create('mode',mode) );
    Parameters.Add( TPair<AnsiString,TValue>.Create('distribution',distribution) );

    if seed <> nil then
        Parameters.Add( TPair<AnsiString,TValue>.Create('seed',seed^) );

    PyInstance := GetKerasClassIstance('initializers.VarianceScaling');
    init();
end;

{ TOrthogonal }

constructor TOrthogonal.Create(gain: Double; seed: PInteger);
begin
    inherited Create;

    Parameters.Add( TPair<AnsiString,TValue>.Create('gain',gain) );

    if seed <> nil then
        Parameters.Add( TPair<AnsiString,TValue>.Create('seed',seed^) );

    PyInstance := GetKerasClassIstance('initializers.Orthogonal');
    init();
end;

{ TIdentity }

constructor TIdentity.Create(gain: Double);
begin
    inherited Create;

    Parameters.Add( TPair<AnsiString,TValue>.Create('gain',gain) );

    PyInstance := GetKerasClassIstance('initializers.Identity');
    init();
end;

{ TLecunUniform }

constructor TLecunUniform.Create(seed: PInteger);
begin
    inherited Create;

    if seed <> nil then
        Parameters.Add( TPair<AnsiString,TValue>.Create('seed',seed^) );

    PyInstance := GetKerasClassIstance('initializers.lecun_uniform');
    init();
end;

{ TGlorotNormal }

constructor TGlorotNormal.Create(seed: PInteger);
begin
    inherited Create;

    if seed <> nil then
        Parameters.Add( TPair<AnsiString,TValue>.Create('seed',seed^) );

    PyInstance := GetKerasClassIstance('initializers.glorot_normal');
    init();
end;

{ TGlorotUniform }

constructor TGlorotUniform.Create(seed: PInteger);
begin
    inherited Create;

    if seed <> nil then
        Parameters.Add( TPair<AnsiString,TValue>.Create('seed',seed^) );

    PyInstance := GetKerasClassIstance('initializers.glorot_uniform');
    init();
end;

{ THeUniform }

constructor THeUniform.Create(seed: PInteger);
begin
    inherited Create;

    if seed <> nil then
        Parameters.Add( TPair<AnsiString,TValue>.Create('seed',seed^) );

    PyInstance := GetKerasClassIstance('initializers.he_uniform');
    init();
end;

{ THeNormal }

constructor THeNormal.Create(seed: PInteger);
begin
    inherited Create;

    if seed <> nil then
        Parameters.Add( TPair<AnsiString,TValue>.Create('seed',seed^) );

    PyInstance := GetKerasClassIstance('initializers.he_normal');
    init();
end;

{ TLecunNormal }

constructor TLecunNormal.Create(seed: PInteger);
begin
    inherited Create;

    if seed <> nil then
        Parameters.Add( TPair<AnsiString,TValue>.Create('seed',seed^) );

    PyInstance := GetKerasClassIstance('initializers.lecun_normal');
    init();
end;

{ TL1L2 }

constructor TL1L2.Create(l1, l2: Double);
begin
    inherited Create;

    Parameters.Add( TPair<AnsiString,TValue>.Create('l1',l1) );
    Parameters.Add( TPair<AnsiString,TValue>.Create('l2',l2) );

    PyInstance := GetKerasClassIstance('regularizers.L1L2');
    init();
end;

{ TL1 }

constructor TL1.Create(l1: Double);
begin
    inherited Create;

    Parameters.Add( TPair<AnsiString,TValue>.Create('l1',l1) );

    PyInstance := GetKerasClassIstance('regularizers.L1');
    init();
end;

{ TL2 }

constructor TL2.Create(l2: Double);
begin
    inherited Create;

    Parameters.Add( TPair<AnsiString,TValue>.Create('l2',l2) );

    PyInstance := GetKerasClassIstance('regularizers.L2');
    init();
end;

{ TSGD }

constructor TSGD.Create(lr, momentum, decay: Double; nesterov: Boolean);
begin
    inherited Create;

    Parameters.Add( TPair<AnsiString,TValue>.Create('lr',lr) );
    Parameters.Add( TPair<AnsiString,TValue>.Create('momentum',momentum) );
    Parameters.Add( TPair<AnsiString,TValue>.Create('decay',decay) );
    Parameters.Add( TPair<AnsiString,TValue>.Create('nesterov',nesterov) );

    PyInstance := GetKerasClassIstance('optimizers.SGD');
    init();
end;

{ TRMSprop }

constructor TRMSprop.Create(lr, rho: Double; epsilon: PDouble; decay: Double);
begin
    inherited Create;

    Parameters.Add( TPair<AnsiString,TValue>.Create('lr',lr) );
    Parameters.Add( TPair<AnsiString,TValue>.Create('rho',rho) );

    if epsilon <> nil then
      Parameters.Add( TPair<AnsiString,TValue>.Create('epsilon',epsilon^) );

    Parameters.Add( TPair<AnsiString,TValue>.Create('decay',decay) );

    PyInstance := GetKerasClassIstance('optimizers.RMSprop');
    init();
end;

{ TAdagrad }

constructor TAdagrad.Create(lr: Double; epsilon: PDouble; decay: Double);
begin
    inherited Create;

    Parameters.Add( TPair<AnsiString,TValue>.Create('lr',lr) );

    if epsilon <> nil then
      Parameters.Add( TPair<AnsiString,TValue>.Create('epsilon',epsilon^) );

    Parameters.Add( TPair<AnsiString,TValue>.Create('decay',decay) );

    PyInstance := GetKerasClassIstance('optimizers.Adagrad');
    init();
end;

{ TAdadelta }

constructor TAdadelta.Create(lr, rho: Double; epsilon: PDouble; decay: Double);
begin
    inherited Create;

    Parameters.Add( TPair<AnsiString,TValue>.Create('lr',lr) );
    Parameters.Add( TPair<AnsiString,TValue>.Create('rho',rho) );

    if epsilon <> nil then
      Parameters.Add( TPair<AnsiString,TValue>.Create('epsilon',epsilon^) );

    Parameters.Add( TPair<AnsiString,TValue>.Create('decay',decay) );

    PyInstance := GetKerasClassIstance('optimizers.Adadelta');
    init();
end;

{ TAdam }

constructor TAdam.Create(lr, beta_1, beta_2: Double; epsilon: PDouble; decay: double; amsgrad: Boolean);
begin
    inherited Create;

    Parameters.Add( TPair<AnsiString,TValue>.Create('lr',lr) );
    Parameters.Add( TPair<AnsiString,TValue>.Create('beta_1',beta_1) );
    Parameters.Add( TPair<AnsiString,TValue>.Create('beta_2',beta_2) );

    if epsilon <> nil then Parameters.Add( TPair<AnsiString,TValue>.Create('epsilon',epsilon^) )
    else                   Parameters.Add( TPair<AnsiString,TValue>.Create('epsilon', TPythonObject.None ));

    Parameters.Add( TPair<AnsiString,TValue>.Create('decay',decay) );
    Parameters.Add( TPair<AnsiString,TValue>.Create('amsgrad',amsgrad) );

    PyInstance := GetKerasClassIstance('optimizers.Adam');
    init();
end;

{ TAdamax }

constructor TAdamax.Create(lr, beta_1, beta_2: Double; epsilon: PDouble; decay: double);
begin
    inherited Create;

    Parameters.Add( TPair<AnsiString,TValue>.Create('lr',lr) );
    Parameters.Add( TPair<AnsiString,TValue>.Create('beta_1',beta_1) );
    Parameters.Add( TPair<AnsiString,TValue>.Create('beta_2',beta_2) );

    if epsilon <> nil then
      Parameters.Add( TPair<AnsiString,TValue>.Create('epsilon',epsilon^) );

    Parameters.Add( TPair<AnsiString,TValue>.Create('decay',decay) );

    PyInstance := GetKerasClassIstance('optimizers.Adamax');
    init();
end;

{ TNadam }

constructor TNadam.Create(lr, beta_1, beta_2: Double; epsilon: PDouble; schedule_decay: double);
begin
    inherited Create;

    Parameters.Add( TPair<AnsiString,TValue>.Create('lr',lr) );
    Parameters.Add( TPair<AnsiString,TValue>.Create('beta_1',beta_1) );
    Parameters.Add( TPair<AnsiString,TValue>.Create('beta_2',beta_2) );

    if epsilon <> nil then
      Parameters.Add( TPair<AnsiString,TValue>.Create('epsilon',epsilon^) );

    Parameters.Add( TPair<AnsiString,TValue>.Create('schedule_decay',schedule_decay) );

    PyInstance := GetKerasClassIstance('optimizers.Nadam');
    init();
end;

{ TMetrics }

constructor TMetrics.Create;
begin
    inherited Create;

    caller  := GetKerasClassIstance('metrics');
end;

function TMetrics.BinaryAccuracy(y_true, y_pred: TNDarray): TNDarray;
begin
    Parameters.Clear;

    Parameters.Add( TPair<AnsiString,TValue>.Create('y_true',y_true) );
    Parameters.Add( TPair<AnsiString,TValue>.Create('y_pred',y_pred) );

    Result := TNDArray.Create( InvokeStaticMethod(caller,'binary_accuracy',Parameters) );

end;

function TMetrics.CategoricalAccuracy(y_true, y_pred: TNDarray): TNDarray;
begin
    Parameters.Clear;

    Parameters.Add( TPair<AnsiString,TValue>.Create('y_true',y_true) );
    Parameters.Add( TPair<AnsiString,TValue>.Create('y_pred',y_pred) );

    Result := TNDArray.Create( InvokeStaticMethod(caller,'categorical_accuracy',Parameters) );
end;

function TMetrics.Cosine(y_true, y_pred: TNDarray): TNDarray;
begin
    Parameters.Clear;

    Parameters.Add( TPair<AnsiString,TValue>.Create('y_true',y_true) );
    Parameters.Add( TPair<AnsiString,TValue>.Create('y_pred',y_pred) );

    Result := TNDArray.Create( InvokeStaticMethod(caller,'cosine',Parameters) )
end;

function TMetrics.MAE(y_true, y_pred: TNDarray): TNDarray;
begin
    Parameters.Clear;

    Parameters.Add( TPair<AnsiString,TValue>.Create('y_true',y_true) );
    Parameters.Add( TPair<AnsiString,TValue>.Create('y_pred',y_pred) );

    Result := TNDArray.Create( InvokeStaticMethod(caller,'mae',Parameters) )
end;

function TMetrics.MAPE(y_true, y_pred: TNDarray): TNDarray;
begin
    Parameters.Clear;

    Parameters.Add( TPair<AnsiString,TValue>.Create('y_true',y_true) );
    Parameters.Add( TPair<AnsiString,TValue>.Create('y_pred',y_pred) );

    Result := TNDArray.Create( InvokeStaticMethod(caller,'mape',Parameters) )
end;

function TMetrics.MSE(y_true, y_pred: TNDarray): TNDarray;
begin
    Parameters.Clear;

    Parameters.Add( TPair<AnsiString,TValue>.Create('y_true',y_true) );
    Parameters.Add( TPair<AnsiString,TValue>.Create('y_pred',y_pred) );

    Result := TNDArray.Create( InvokeStaticMethod(caller,'mse',Parameters) )
end;

function TMetrics.MSLE(y_true, y_pred: TNDarray): TNDarray;
begin
    Parameters.Clear;

    Parameters.Add( TPair<AnsiString,TValue>.Create('y_true',y_true) );
    Parameters.Add( TPair<AnsiString,TValue>.Create('y_pred',y_pred) );

    Result := TNDArray.Create( InvokeStaticMethod(caller,'msle',Parameters) )
end;

function TMetrics.SparseCategoricalAccuracy(y_true, y_pred: TNDarray): TNDarray;
begin
    Parameters.Clear;

    Parameters.Add( TPair<AnsiString,TValue>.Create('y_true',y_true) );
    Parameters.Add( TPair<AnsiString,TValue>.Create('y_pred',y_pred) );

    Result := TNDArray.Create( InvokeStaticMethod(caller,'sparse_categorical_accuracy',Parameters) )
end;

function TMetrics.SparseTopKCategoricalAccuracy(y_true, y_pred: TNDarray; k: Integer): TNDarray;
begin
    Parameters.Clear;

    Parameters.Add( TPair<AnsiString,TValue>.Create('y_true',y_true) );
    Parameters.Add( TPair<AnsiString,TValue>.Create('y_pred',y_pred) );

    Result := TNDArray.Create( InvokeStaticMethod(caller,'sparse_top_k_categorical_accuracy',Parameters) )
end;

function TMetrics.TopKCategoricalAccuracy(y_true, y_pred: TNDarray; k: Integer): TNDarray;
begin
    Parameters.Clear;

    Parameters.Add( TPair<AnsiString,TValue>.Create('y_true',y_true) );
    Parameters.Add( TPair<AnsiString,TValue>.Create('y_pred',y_pred) );

    Result := TNDArray.Create( InvokeStaticMethod(caller,'top_k_categorical_accuracy',Parameters) )
end;

{ TLosses }

constructor TLosses.Create;
begin
    inherited Create;

    caller  := GetKerasClassIstance('losses');
end;


function TLosses.BinaryCrossentropy(y_true, y_pred: TNDarray): TNDarray;
begin
    Parameters.Clear;

    Parameters.Add( TPair<AnsiString,TValue>.Create('y_true',y_true) );
    Parameters.Add( TPair<AnsiString,TValue>.Create('y_pred',y_pred) );

    Result := TNDArray.Create( InvokeStaticMethod(caller,'binary_crossentropy',Parameters) )
end;

function TLosses.CategoricalCrossentropy(y_true, y_pred: TNDarray): TNDarray;
begin
    Parameters.Clear;

    Parameters.Add( TPair<AnsiString,TValue>.Create('y_true',y_true) );
    Parameters.Add( TPair<AnsiString,TValue>.Create('y_pred',y_pred) );

    Result := TNDArray.Create( InvokeStaticMethod(caller,'categorical_crossentropy',Parameters) )
end;

function TLosses.CategoricalHinge(y_true, y_pred: TNDarray): TNDarray;
begin
    Parameters.Clear;

    Parameters.Add( TPair<AnsiString,TValue>.Create('y_true',y_true) );
    Parameters.Add( TPair<AnsiString,TValue>.Create('y_pred',y_pred) );

    Result := TNDArray.Create( InvokeStaticMethod(caller,'categorical_hinge',Parameters) )
end;

function TLosses.CosineProximity(y_true, y_pred: TNDarray): TNDarray;
begin
    Parameters.Clear;

    Parameters.Add( TPair<AnsiString,TValue>.Create('y_true',y_true) );
    Parameters.Add( TPair<AnsiString,TValue>.Create('y_pred',y_pred) );

    Result := TNDArray.Create( InvokeStaticMethod(caller,'cosine_proximity',Parameters) )
end;

function TLosses.Hinge(y_true, y_pred: TNDarray): TNDarray;
begin
    Parameters.Clear;

    Parameters.Add( TPair<AnsiString,TValue>.Create('y_true',y_true) );
    Parameters.Add( TPair<AnsiString,TValue>.Create('y_pred',y_pred) );

    Result := TNDArray.Create( InvokeStaticMethod(caller,'hinge',Parameters) )
end;

function TLosses.KullbackLeiblerDivergence(y_true, y_pred: TNDarray): TNDarray;
begin
    Parameters.Clear;

    Parameters.Add( TPair<AnsiString,TValue>.Create('y_true',y_true) );
    Parameters.Add( TPair<AnsiString,TValue>.Create('y_pred',y_pred) );

    Result := TNDArray.Create( InvokeStaticMethod(caller,'kullback_leibler_divergence',Parameters) )
end;

function TLosses.LogCosh(y_true, y_pred: TNDarray): TNDarray;
begin
    Parameters.Clear;

    Parameters.Add( TPair<AnsiString,TValue>.Create('y_true',y_true) );
    Parameters.Add( TPair<AnsiString,TValue>.Create('y_pred',y_pred) );

    Result := TNDArray.Create( InvokeStaticMethod(caller,'logcosh',Parameters) )
end;

function TLosses.MeanAbsoluteError(y_true, y_pred: TNDarray): TNDarray;
begin
    Parameters.Clear;

    Parameters.Add( TPair<AnsiString,TValue>.Create('y_true',y_true) );
    Parameters.Add( TPair<AnsiString,TValue>.Create('y_pred',y_pred) );

    Result := TNDArray.Create( InvokeStaticMethod(caller,'mean_absolute_error',Parameters) )
end;

function TLosses.MeanAbsolutePercentageError(y_true, y_pred: TNDarray): TNDarray;
begin
    Parameters.Clear;

    Parameters.Add( TPair<AnsiString,TValue>.Create('y_true',y_true) );
    Parameters.Add( TPair<AnsiString,TValue>.Create('y_pred',y_pred) );

    Result := TNDArray.Create( InvokeStaticMethod(caller,'mean_absolute_percentage_error',Parameters) )
end;

function TLosses.MeanSquaredError(y_true, y_pred: TNDarray): TNDarray;
begin
    Parameters.Clear;

    Parameters.Add( TPair<AnsiString,TValue>.Create('y_true',y_true) );
    Parameters.Add( TPair<AnsiString,TValue>.Create('y_pred',y_pred) );

    Result := TNDArray.Create( InvokeStaticMethod(caller,'mean_squared_error',Parameters) )
end;

function TLosses.MeanSquaredLogarithmicError(y_true, y_pred: TNDarray): TNDarray;
begin
    Parameters.Clear;

    Parameters.Add( TPair<AnsiString,TValue>.Create('y_true',y_true) );
    Parameters.Add( TPair<AnsiString,TValue>.Create('y_pred',y_pred) );

    Result := TNDArray.Create( InvokeStaticMethod(caller,'mean_squared_logarithmic_error',Parameters) )
end;

function TLosses.Poisson(y_true, y_pred: TNDarray): TNDarray;
begin
    Parameters.Clear;

    Parameters.Add( TPair<AnsiString,TValue>.Create('y_true',y_true) );
    Parameters.Add( TPair<AnsiString,TValue>.Create('y_pred',y_pred) );

    Result := TNDArray.Create( InvokeStaticMethod(caller,'poisson',Parameters) )
end;

function TLosses.SparseCategoricalCrossentropy(y_true, y_pred: TNDarray): TNDarray;
begin
    Parameters.Clear;

    Parameters.Add( TPair<AnsiString,TValue>.Create('y_true',y_true) );
    Parameters.Add( TPair<AnsiString,TValue>.Create('y_pred',y_pred) );

    Result := TNDArray.Create( InvokeStaticMethod(caller,'sparse_categorical_crossentropy',Parameters) )
end;

function TLosses.SquaredHinge(y_true, y_pred: TNDarray): TNDarray;
begin
    Parameters.Clear;

    Parameters.Add( TPair<AnsiString,TValue>.Create('y_true',y_true) );
    Parameters.Add( TPair<AnsiString,TValue>.Create('y_pred',y_pred) );

    Result := TNDArray.Create( InvokeStaticMethod(caller,'squared_hinge',Parameters) )
end;

{ TBostonHousing }

constructor TBostonHousing.Create;
begin
    inherited Create;
end;

function TBostonHousing.load_data(path: string ; test_split: double; seed : Integer): TArray<TNDArray>;
var
    py     : TPythonObject;
begin
    Parameters.Add( TPair<AnsiString,TValue>.Create('path',path) );
    Parameters.Add( TPair<AnsiString,TValue>.Create('test_split',test_split) );
    Parameters.Add( TPair<AnsiString,TValue>.Create('seed',seed) );

    PyInstance := GetKerasClassIstance('datasets.boston_housing');

    py  := InvokeStaticMethod(PyInstance,'load_data',Parameters);

    Result := TTupleSolver.TupleToList(py);

end;

{ TCifar10 }

constructor TCifar10.Create;
begin
    inherited Create;
end;

class function TCifar10.load_data: TArray<TNDArray>;
var
    py     : TPythonObject;
begin
    Create;

    var args: TList< TPair<AnsiString,TValue> > := TList< TPair<AnsiString,TValue> >.Create;

    py         := InvokeStaticMethod( GetKerasClassIstance('datasets.cifar10') ,'load_data',args);
    Result     := TTupleSolver.TupleToList(py);

end;

{ TCifar100 }

constructor TCifar100.Create;
begin
    inherited Create;
end;

class function TCifar100.load_data(label_mode: string): TArray<TNDArray>;
var
    py     : TPythonObject;
begin
    Create;

    var args: TList< TPair<AnsiString,TValue> > := TList< TPair<AnsiString,TValue> >.Create;

    args.Add( TPair<AnsiString,TValue>.Create('label_mode',label_mode) );

    py         := InvokeStaticMethod( GetKerasClassIstance('datasets.cifar100') ,'load_data',args);
    Result     := TTupleSolver.TupleToList(py);

end;

{ TFashionMNIST }

constructor TFashionMNIST.Create;
begin
    inherited Create;
end;

class function TFashionMNIST.load_data: TArray<TNDArray>;
var
    py     : TPythonObject;
begin
    Create;

    var args: TList< TPair<AnsiString,TValue> > := TList< TPair<AnsiString,TValue> >.Create;

    py         := InvokeStaticMethod( GetKerasClassIstance('datasets.fashion_mnist') ,'load_data',args);
    Result     := TTupleSolver.TupleToList(py);

end;

{ TMNIST }

constructor TMNIST.Create;
begin
    inherited Create;
end;

class function TMNIST.load_data(path: string): TArray<TNDArray>;
var
    py     : TPythonObject;
begin
    Create;

    var args: TList< TPair<AnsiString,TValue> > := TList< TPair<AnsiString,TValue> >.Create;

    args.Add( TPair<AnsiString,TValue>.Create('path',path) );

    py         := InvokeStaticMethod(GetKerasClassIstance('datasets.mnist'),'load_data',args);
    Result     := TTupleSolver.TupleToList(py);

end;

{ TIMDB }

constructor TIMDB.Create;
begin
    inherited Create;
end;

function TIMDB.GetWordIndex(path: string): TDictionary<string, Integer>;
var
  dict : TDictionary<string, Integer>;
  py   : TPyDict;
  keys : TArray<string>;
  key  : string;
begin
    dict := TDictionary<string, Integer>.Create;

    Parameters.Clear;

    Parameters.Add( TPair<AnsiString,TValue>.Create('path',path) );

    PyInstance := GetKerasClassIstance('datasets.imdb');
    py := TPyDict.Create( InvokeStaticMethod(PyInstance,'get_word_index',Parameters) );

    keys := py.Keys.AsArrayofString;
    for key in keys do
       dict.AddOrSetValue(key,py[key].AsInteger);

    Result := dict;
end;

function TIMDB.load_data(path: string; num_words: PInteger; skip_top: Integer; maxlen: PInteger; seed, start_char,
                                                                                   oov_char, index_from: Integer): TArray<TNDArray>;
var
    py     : TPythonObject;
begin
    Parameters.Clear;

    Parameters.Add( TPair<AnsiString,TValue>.Create('path',path) );
    if num_words <> nil then
      Parameters.Add( TPair<AnsiString,TValue>.Create('num_words',num_words^) );
    Parameters.Add( TPair<AnsiString,TValue>.Create('skip_top',skip_top) );
    if num_words <> nil then
      Parameters.Add( TPair<AnsiString,TValue>.Create('maxlen',maxlen^) );
    Parameters.Add( TPair<AnsiString,TValue>.Create('seed',seed) );
    Parameters.Add( TPair<AnsiString,TValue>.Create('start_char',start_char) );
    Parameters.Add( TPair<AnsiString,TValue>.Create('oov_char',oov_char) );
    Parameters.Add( TPair<AnsiString,TValue>.Create('index_from',index_from) );

    PyInstance := GetKerasClassIstance('datasets.imdb');

    py  := InvokeStaticMethod(PyInstance,'load_data',Parameters);

    Result := TTupleSolver.TupleToList(py);

end;

{ TReuters }

constructor TReuters.Create;
begin
    inherited Create;
end;

function TReuters.load_data(path: string; num_words: PInteger; skip_top: Integer; maxlen: PInteger; test_split: Double;
                                                       seed, start_char, oov_char, index_from: Integer): TArray<TNDArray>;
var
    py     : TPythonObject;
begin
    Parameters.Clear;

    Parameters.Add( TPair<AnsiString,TValue>.Create('path',path) );
    if num_words <> nil then
      Parameters.Add( TPair<AnsiString,TValue>.Create('num_words',num_words^) );
    Parameters.Add( TPair<AnsiString,TValue>.Create('skip_top',skip_top) );
    if num_words <> nil then
      Parameters.Add( TPair<AnsiString,TValue>.Create('maxlen',maxlen^) );
    Parameters.Add( TPair<AnsiString,TValue>.Create('test_split',test_split) );
    Parameters.Add( TPair<AnsiString,TValue>.Create('seed',seed) );
    Parameters.Add( TPair<AnsiString,TValue>.Create('start_char',start_char) );
    Parameters.Add( TPair<AnsiString,TValue>.Create('oov_char',oov_char) );
    Parameters.Add( TPair<AnsiString,TValue>.Create('index_from',index_from) );

    PyInstance := GetKerasClassIstance('datasets.reuters');

    py  := InvokeStaticMethod(PyInstance,'load_data',Parameters);

    Result := TTupleSolver.TupleToList(py);


end;

{ TMaxNorm }

constructor TMaxNorm.Create(max_value: double; axis: Integer);
begin
    inherited Create;

    Parameters.Add( TPair<AnsiString,TValue>.Create('max_value',max_value) );
    Parameters.Add( TPair<AnsiString,TValue>.Create('axis',axis) );

    PyInstance := GetKerasClassIstance('constraints.MaxNorm');
    init();
end;

{ TNonNeg }

constructor TNonNeg.Create;
begin
    inherited Create;

    PyInstance := GetKerasClassIstance('constraints.NonNeg');
    init();
end;

{ TUnitNorm }

constructor TUnitNorm.Create(axis: Integer);
begin
    inherited Create;

    Parameters.Add( TPair<AnsiString,TValue>.Create('axis',axis) );

    PyInstance := GetKerasClassIstance('constraints.UnitNorm');
    init();
end;

{ TMinMaxNorm }

constructor TMinMaxNorm.Create(min_value, max_value, rate: Double; axis: Integer);
begin
    inherited Create;

    Parameters.Add( TPair<AnsiString,TValue>.Create('min_value',min_value) );
    Parameters.Add( TPair<AnsiString,TValue>.Create('max_value',max_value) );
    Parameters.Add( TPair<AnsiString,TValue>.Create('rate',rate) );
    Parameters.Add( TPair<AnsiString,TValue>.Create('axis',axis) );

    PyInstance := GetKerasClassIstance('constraints.MinMaxNorm');
    init();
end;

{ TCallback }

constructor TCallback.Create;
begin
    inherited Create;

    pyInstance := GetKerasClassIstance('callbacks.Callback');
end;

constructor TCallback.Create(py: PPyObject);
begin
    inherited Create;

    pyInstance := TPythonObject.Create(py)
end;

constructor TCallback.Create(py: TPythonObject);
begin
    inherited Create;

    pyInstance := py
end;

class function TCallback.Custom(name, fileOrcode: string; isFile: Boolean): TCallback;
var
 code : string;
 py   : TPythonObject;
begin
   if isFile then code := TFile.ReadAllText(fileOrcode)
   else           code := fileOrcode;

   py := TPythonObject.ModuleFromString(name,code);
   py := py.InvokeMethod(name);
   Result := TCallback.Create(py);
end;

function TCallback.GetDoubleArray(prop: string): TArray<Double>;
begin
    Result := PyInstance.GetAttr(prop).AsArrayofDouble;
end;

{ TBaseLogger }

constructor TBaseLogger.Create(stateful_metrics: TArray<string>);
begin
    inherited Create;

    Parameters.Add( TPair<AnsiString,TValue>.Create('stateful_metrics',TValue.FromArray<String>(stateful_metrics)) );


    PyInstance := GetKerasClassIstance('callbacks.BaseLogger');
    init();
end;


{ TTerminateOnNaN }

constructor TTerminateOnNaN.Create;
begin
    inherited Create;

    PyInstance := GetKerasClassIstance('callbacks.TerminateOnNaN');
    init();
end;

{ TProgbarLogger }

constructor TProgbarLogger.Create(count_mode: string; stateful_metrics: TArray<string>);
begin
    inherited Create;

    Parameters.Add( TPair<AnsiString,TValue>.Create('count_mode',count_mode ));
    Parameters.Add( TPair<AnsiString,TValue>.Create('stateful_metrics',TValue.FromArray<String>(stateful_metrics)) );

    PyInstance := GetKerasClassIstance('callbacks.ProgbarLogger');
    init();
end;

{ THistory }

constructor THistory.Create;
begin
    inherited Create;

    PyInstance := GetKerasClassIstance('callbacks.History');
    init();
end;

constructor THistory.Create(py: PPyObject);
begin
    inherited Create;

    pyInstance := TPythonObject.Create(py)
end;

constructor THistory.Create(py: TPythonObject);
begin
    inherited Create;

    pyInstance := py
end;

function THistory.GetEpoch: TArray<Integer>;
begin
     Result := PyInstance.GetAttr('epoch').AsArrayofInt
end;

function THistory.GetHistoryLogs: TDictionary<string, TArray<Double>>;
var
  dict : TDictionary<string, TArray<Double>>;
  py   : TPyDict;
  keys : TArray<string>;
  key  : string;
begin
    dict := TDictionary<string, TArray<Double> >.Create;

    py := TPyDict.Create( PyInstance.GetAttr('history') );

    keys := py.Keys.AsArrayofString;

    for key in keys do
       dict.AddOrSetValue(key,py[key].AsArrayofDouble);

    Result := dict;

end;

{ TModelCheckpoint }

constructor TModelCheckpoint.Create(filepath, monitor: string; verbose: Integer; save_best_only,
                                                        save_weights_only: Boolean; mode: string; period: Integer);
begin
    inherited Create;

    Parameters.Add( TPair<AnsiString,TValue>.Create('filepath',filepath ));
    Parameters.Add( TPair<AnsiString,TValue>.Create('monitor',monitor ));
    Parameters.Add( TPair<AnsiString,TValue>.Create('verbose',verbose ));
    Parameters.Add( TPair<AnsiString,TValue>.Create('save_best_only',save_best_only ));
    Parameters.Add( TPair<AnsiString,TValue>.Create('save_weights_only',save_weights_only ));
    Parameters.Add( TPair<AnsiString,TValue>.Create('mode',mode ));
    Parameters.Add( TPair<AnsiString,TValue>.Create('period',period ));

    PyInstance := GetKerasClassIstance('callbacks.ModelCheckpoint');
    init();
end;

{ TEarlyStopping }

constructor TEarlyStopping.Create(monitor: string; min_delta: Double; patience, verbose: Integer; mode: string;
                                                                     baseline: PDouble; restore_best_weights: Boolean);
begin
    inherited Create;

    Parameters.Add( TPair<AnsiString,TValue>.Create('monitor',monitor ));
    Parameters.Add( TPair<AnsiString,TValue>.Create('min_delta',min_delta ));
    Parameters.Add( TPair<AnsiString,TValue>.Create('patience',patience ));
    Parameters.Add( TPair<AnsiString,TValue>.Create('verbose',verbose ));
    Parameters.Add( TPair<AnsiString,TValue>.Create('mode',mode ));
    if baseline <> nil then
      Parameters.Add( TPair<AnsiString,TValue>.Create('baseline',baseline^ ));
    Parameters.Add( TPair<AnsiString,TValue>.Create('restore_best_weights',restore_best_weights ));

    PyInstance := GetKerasClassIstance('callbacks.EarlyStopping');
    init();
end;

{ TRemoteMonitor }

constructor TRemoteMonitor.Create(root, path, field: string; headers: TDictionary<string, string>; send_as_json: Boolean);
begin
    inherited Create;

    Parameters.Add( TPair<AnsiString,TValue>.Create('root',root ));
    Parameters.Add( TPair<AnsiString,TValue>.Create('path',path ));
    Parameters.Add( TPair<AnsiString,TValue>.Create('field',field ));
    if headers <> nil then
       Parameters.Add( TPair<AnsiString,TValue>.Create('headers', ToDict(headers) ));
    Parameters.Add( TPair<AnsiString,TValue>.Create('send_as_json',send_as_json ));

    PyInstance := GetKerasClassIstance('callbacks.RemoteMonitor');
    init();
end;

{ TLearningRateScheduler }

{ TODO -oMax -c : verificare ! errore anche nel progetto originale 25/02/2020 19:12:54 }
constructor TLearningRateScheduler.Create(schedule: TFunSchedule; verbose: Integer);
begin
    inherited Create;

    Parameters.Add( TPair<AnsiString,TValue>.Create('schedule',@schedule ));
    Parameters.Add( TPair<AnsiString,TValue>.Create('verbose',verbose ));

    PyInstance := GetKerasClassIstance('callbacks.LearningRateScheduler');
    init();
end;

{ TTensorBoard }

constructor TTensorBoard.Create(log_dir: string; histogram_freq, batch_size: Integer; write_graph, write_grads,
                                write_images: Boolean; embeddings_freq: Integer; embeddings_layer_names: TArray<String>;
                                embeddings_metadata: TDictionary<string, string>; embeddings_data: TNDarray; update_freq: string);
begin
    inherited Create;

    Parameters.Add( TPair<AnsiString,TValue>.Create('log_dir',log_dir ));
    Parameters.Add( TPair<AnsiString,TValue>.Create('histogram_freq',histogram_freq ));
    Parameters.Add( TPair<AnsiString,TValue>.Create('batch_size',batch_size ));
    Parameters.Add( TPair<AnsiString,TValue>.Create('write_graph',write_graph ));
    Parameters.Add( TPair<AnsiString,TValue>.Create('write_grads',write_grads ));
    Parameters.Add( TPair<AnsiString,TValue>.Create('write_images',write_images ));
    Parameters.Add( TPair<AnsiString,TValue>.Create('embeddings_freq',embeddings_freq ));
    if embeddings_layer_names <> nil then
        Parameters.Add( TPair<AnsiString,TValue>.Create('embeddings_layer_names',TValue.FromArray<string>(embeddings_layer_names) ));
    if embeddings_metadata <> nil then
        Parameters.Add( TPair<AnsiString,TValue>.Create('embeddings_metadata',ToDict(embeddings_metadata) ));
    if embeddings_data <> nil then
        Parameters.Add( TPair<AnsiString,TValue>.Create('embeddings_data', embeddings_data ));
    Parameters.Add( TPair<AnsiString,TValue>.Create('update_freq',update_freq ));

    PyInstance := GetKerasClassIstance('callbacks.TensorBoard');
    init();
end;

{ TReduceLROnPlateau }

constructor TReduceLROnPlateau.Create(monitor: string; factor: Double; patience, verbose: Integer; mode: string;
                                                                 min_delta: Double; cooldown: Integer; min_lr: Double);
begin
    inherited Create;

    Parameters.Add( TPair<AnsiString,TValue>.Create('monitor',monitor ));
    Parameters.Add( TPair<AnsiString,TValue>.Create('factor',factor ));
    Parameters.Add( TPair<AnsiString,TValue>.Create('patience',patience ));
    Parameters.Add( TPair<AnsiString,TValue>.Create('verbose',verbose ));
    Parameters.Add( TPair<AnsiString,TValue>.Create('mode',mode ));
    Parameters.Add( TPair<AnsiString,TValue>.Create('min_delta',min_delta ));
    Parameters.Add( TPair<AnsiString,TValue>.Create('cooldown',cooldown ));
    Parameters.Add( TPair<AnsiString,TValue>.Create('min_lr',min_lr ));

    PyInstance := GetKerasClassIstance('callbacks.ReduceLROnPlateau');
    init();
end;

{ TCSVLogger }

constructor TCSVLogger.Create(filename, separator: string; append: Boolean);
begin
    inherited Create;

    Parameters.Add( TPair<AnsiString,TValue>.Create('filename',filename ));
    Parameters.Add( TPair<AnsiString,TValue>.Create('separator',separator ));
    Parameters.Add( TPair<AnsiString,TValue>.Create('append',append ));

    PyInstance := GetKerasClassIstance('callbacks.CSVLogger');
    init();
end;

{ TBackend }

constructor TBackend.Create;
begin
    inherited Create;

    caller  := GetKerasClassIstance('backend');
end;

function TBackend.Cast(x: TPythonObject; dtype: string): TPythonObject;
begin
    Parameters.Clear;

    Parameters.Add( TPair<AnsiString,TValue>.Create('x',x) );
    Parameters.Add( TPair<AnsiString,TValue>.Create('dtype',dtype) );

    Result := TPythonObject.Create( InvokeStaticMethod(caller,'cast',Parameters) )
end;

function TBackend.CastToFloatX(x: TNDarray): TNDarray;
begin
    Parameters.Clear;

    Parameters.Add( TPair<AnsiString,TValue>.Create('x',x) );

    Result := TNDarray.Create( InvokeStaticMethod(caller,'cast_to_floatx',Parameters) )
end;

procedure TBackend.ClearSession;
begin
    Parameters.Clear;

    InvokeStaticMethod(caller,'clear_session',Parameters)
end;

procedure TBackend.DisableEager;
begin
    Parameters.Clear;

    InvokeStaticMethod( GetTFClassIstance('compat.v1'),'disable_eager_execution',Parameters )
end;

function TBackend.Epsilon: Double;
begin
    Parameters.Clear;

    Result := InvokeStaticMethod(caller,'epsilon',Parameters).AsDouble;
end;

function TBackend.FloatX: string;
begin
    Parameters.Clear;

    Result := InvokeStaticMethod(caller,'floatx',Parameters).ToString;
end;

function TBackend.GetBackend: string;
begin
    Parameters.Clear;

    Result := InvokeStaticMethod(caller,'backend',Parameters).ToString;
end;

function TBackend.GetUid(prefix: string): string;
begin
    Parameters.Clear;

    Parameters.Add( TPair<AnsiString,TValue>.Create('prefix',prefix) );

    Result := InvokeStaticMethod(caller,'get_uid',Parameters).ToString;
end;

function TBackend.ImageDataFormat: string;
begin
    Parameters.Clear;

    Result := InvokeStaticMethod(caller,'image_data_format',Parameters).ToString;
end;

function TBackend.IsSparse(tensor: TNDarray): Boolean;
begin
    Parameters.Clear;

    Parameters.Add( TPair<AnsiString,TValue>.Create('tensor',tensor) );

    Result := InvokeStaticMethod(caller,'is_sparse',Parameters).AsBoolean;
end;

procedure TBackend.ResetUids;
begin
    Parameters.Clear;

    InvokeStaticMethod(caller,'reset_uids',Parameters)
end;

procedure TBackend.SetEpsilon(e: Double);
begin
    Parameters.Clear;

    Parameters.Add( TPair<AnsiString,TValue>.Create('e',e) );

    InvokeStaticMethod(caller,'set_epsilon',Parameters);
end;

procedure TBackend.SetFloatX(floatx: string);
begin
    Parameters.Clear;

    Parameters.Add( TPair<AnsiString,TValue>.Create('floatx',floatx) );

    InvokeStaticMethod(caller,'set_floatx',Parameters);
end;

procedure TBackend.SetImageDataFormat(data_format: string);
begin
    Parameters.Clear;

    Parameters.Add( TPair<AnsiString,TValue>.Create('data_format',data_format) );

    InvokeStaticMethod(caller,'set_image_data_format',Parameters);
end;

function TBackend.ToDense(tensor: TNDarray): TNDarray;
begin
    Parameters.Clear;

    Parameters.Add( TPair<AnsiString,TValue>.Create('tensor',tensor) );

    Result := TNDarray.Create( InvokeStaticMethod(caller,'to_dense',Parameters) )
end;

end.
