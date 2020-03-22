{*******************************************************}
{                                                       }
{       Keras Wrapper for Delphi                        }
{                                                       }
{       Copyright (C) 2020 Pigrecos(Max)                }
{                                                       }
{*******************************************************}

unit Keras.Models;

interface
   uses  System.SysUtils, System.Generics.Collections,
         PythonEngine, Keras,np.Models,Python.Utils,
         Keras.Layers;

type

  // BaseModel

  TBaseModel = class(TBase)
     public
       constructor Create;
       procedure Compile(optimizer          : TStringOrInstance;
                         loss               : string;
                         metrics            : TArray<string>= nil;
                         loss_weights       : TArray<Double> = nil;
                         sample_weight_mode : PChar   = nil;
                         weighted_metrics   : TArray<string> = nil;
                         target_tensors     : TArray<TNDarray>= nil);
       function Fit(x                : TNDarray;
                    y                : TNDarray;
                    batch_size       : PInteger = nil;
                    epochs           : Integer = 1;
                    verbose          : Integer= 1;
                    callbacks        : TArray<TCallback> = nil;
                    validation_split : Double = 0.0;
                    validation_data  : TArray<TNDarray> = nil;
                    shuffle          : Boolean = true;
                    class_weight     : TDictionary<Integer, Double> = nil;
                    sample_weight    : TNDarray = nil;
                    initial_epoch    : Integer = 0;
                    steps_per_epoch  : PInteger = nil;
                    validation_steps : PInteger= nil):THistory;overload;
       function Fit(x                : TNDarray;
                    y                : TNDarray;
                    batch_size       : PInteger ;
                    epochs           : Integer;
                    verbose          : Integer;
                    validation_data  : TArray<TNDarray> ):THistory;overload;
       function Evaluate(x             : TNDarray;
                         y             : TNDarray;
                         batch_size    : PInteger = nil;
                         verbose       : Integer= 1;
                         sample_weight : TNDarray = nil;
                         steps         : PInteger= nil;
                         callbacks     : TArray<TCallback> = nil): TArray<Double>;
       function Predict(x            : TNDarray;
                        batch_size   : PInteger = nil;
                        verbose      : Integer= 1;
                        steps        : PInteger= nil;
                        callbacks    : TArray<TCallback> = nil):TNDarray; overload;
       function Predict(x            : TArray<TNDarray>; // List<NDarray>
                        batch_size   : PInteger = nil;
                        verbose      : Integer= 1;
                        steps        : PInteger= nil;
                        callbacks    : TArray<TCallback> = nil):TNDarray;overload;
       function TrainOnBatch(x             : TNDarray;
                             y             : TNDarray;
                             sample_weight : TNDarray = nil;
                             class_weight  : TDictionary<Integer, Double> = nil): TArray<Double>;
       function TestOnBatch(x             : TNDarray;
                            y             : TNDarray;
                            sample_weight : TNDarray = nil): TArray<Double>;
       function PredictOnBatch(x: TNDarray):TNDarray;
       function FitGenerator(generator           : TSequence;
                             steps_per_epoch     : PInteger = nil;
                             epochs              : Integer = 1;
                             verbose             : Integer= 1;
                             callbacks           : TArray<TCallback> = nil;
                             validation_data     : TSequence = nil;
                             validation_steps    : PInteger = nil;
                             validation_freq     : Integer = 1;
                             class_weight        : TDictionary<Integer, Double> = nil;
                             max_queue_size      : Integer= 10;
                             workers             : Integer = 1;
                             use_multiprocessing : Boolean = false;
                             shuffle             : Boolean= true;
                             initial_epoch       : Integer= 0):THistory;
       function EvaluateGenerator(generator           : TSequence;
                                  steps               : PInteger= nil;
                                  callbacks           : TArray<TCallback> = nil;
                                  max_queue_size      : Integer = 10;
                                  workers             : Integer = 1;
                                  use_multiprocessing : Boolean = false;
                                  verbose             : Integer = 0): TArray<Double>;
       function PredictGenerator(generator           : TSequence;
                                  steps               : PInteger= nil;
                                  callbacks           : TArray<TCallback> = nil;
                                  max_queue_size      : Integer = 10;
                                  workers             : Integer = 1;
                                  use_multiprocessing : Boolean = false;
                                  verbose             : Integer = 0):TNDarray;
       function  ToJson: string;
       procedure SaveWeight(path: string);
       procedure Save(filepath: string; overwrite : Boolean= true; include_optimizer: Boolean = true) ;
       function  GetWeights: TArray<TNDarray>;
       procedure SetWeights(weights: TArray<TNDarray>);
       procedure LoadWeight(path: string);
       procedure Summary(line_length: PInteger = nil; positions : TArray<Double>= nil) ;
       function  ModelFromYaml(Yaml_string:string): TBaseModel;
       procedure SaveOnnx(filePath: string);
       procedure SaveTensorflowJSFormat(artifacts_dir: string; quantize : Boolean= false);

       class function  LoadModel(filepath: string; custom_objects: TDictionary<string, string> = nil; compile : Boolean = true): TBaseModel;
       class function  ModelFromJson(json_string: string): TBaseModel;
  end;

  TModel = class(TBaseModel)
     public
       constructor Create; overload;
       constructor Create(py: PPyObject); overload;
       constructor Create(py: TPythonObject); overload;
       constructor Create(input,output: TArray<TBaseLayer>); overload;
       constructor Create(input: TArray<TBaseLayer>); overload;
  end;

  TSequential = class(TBaseModel)
     public
       constructor Create; overload;
       constructor Create(py: PPyObject); overload;
       constructor Create(py: TPythonObject); overload;
       constructor Create(input: TArray<TBaseLayer>); overload;

       procedure Add(layer: TBaseLayer);
  end;

  // Utils
  TUtil = class(TBase)
     public
       caller : TPythonObject;
       constructor Create;
       function  MultiGPUModel(model: TBaseModel; gpus: TArray<Integer>; cpu_merge: Boolean = true; cpu_relocation: Boolean = false): TBaseModel;overload;
       function  MultiGPUModel(model: TBaseModel; gpus: Integer;         cpu_merge: Boolean = true; cpu_relocation: Boolean = false):TBaseModel; overload;
       function  ToCategorical(y: TNDarray; num_classes : PInteger= nil; dtype: string = 'float32'):TNDarray;
       function  Normalize(y: TNDarray; axis: Integer = -1; order: Integer = 2): TNDarray;
       procedure PlotModel(model: TBaseModel; to_file: string = 'model.png'; show_shapes: Boolean = false; show_layer_names: Boolean = true; rankdir: string = 'TB'; expand_nested: Boolean = false; dpi: Integer = 96) ;
       procedure ConfigTensorFlowBackend(intra_op_parallelism_threads: Integer; inter_op_parallelism_threads: Integer; allow_soft_placement: Boolean; cpu_device_count: Integer;gpu_device_count: Integer );
       procedure device( device_name: string );
 end;

implementation
    uses Winapi.Windows, System.IOUtils, System.Rtti,np.Utils,np.Base, np.Api;

{ TUtil }

constructor TUtil.Create;
begin
    inherited create;

    caller := GetKerasClassIstance('utils');
end;

function TUtil.MultiGPUModel(model: TBaseModel; gpus: TArray<Integer>; cpu_merge: Boolean = true; cpu_relocation: Boolean = false): TBaseModel;
begin
    Result := TBaseModel.Create;

    Parameters.Clear;
    Parameters.Add( TPair<String,TValue>.Create('model',model) );
    Parameters.Add( TPair<String,TValue>.Create('gpus', TValue.FromArray<Integer>(gpus)) );
    Parameters.Add( TPair<String,TValue>.Create('cpu_merge',cpu_merge) );
    Parameters.Add( TPair<String,TValue>.Create('cpu_relocation',cpu_relocation) );

    Result.PyInstance :=  InvokeStaticMethod(caller,'multi_gpu_model',Parameters)
end;

function TUtil.MultiGPUModel(model: TBaseModel; gpus: Integer;cpu_merge: Boolean = true; cpu_relocation: Boolean = false):TBaseModel;
begin
    Result := TBaseModel.Create;

    Parameters.Clear;
    Parameters.Add( TPair<String,TValue>.Create('model',model) );
    Parameters.Add( TPair<String,TValue>.Create('gpus',gpus) );
    Parameters.Add( TPair<String,TValue>.Create('cpu_merge',cpu_merge) );
    Parameters.Add( TPair<String,TValue>.Create('cpu_relocation',cpu_relocation) );

    Result.PyInstance :=  InvokeStaticMethod(caller,'multi_gpu_model',Parameters)
end;

function TUtil.ToCategorical(y: TNDarray; num_classes : PInteger= nil; dtype: string = 'float32'):TNDarray;
begin
    Parameters.Clear;
    Parameters.Add( TPair<String,TValue>.Create('y',y) );

    if num_classes <> nil then Parameters.Add( TPair<String,TValue>.Create('num_classes',num_classes^))
    else                       Parameters.Add( TPair<String,TValue>.Create('num_classes', TPythonObject.None ));

    Parameters.Add( TPair<String,TValue>.Create('dtype',dtype) );

    Result := TNDArray.Create( InvokeStaticMethod(caller,'to_categorical',Parameters) )

end;

function TUtil.Normalize(y: TNDarray; axis: Integer = -1; order: Integer = 2): TNDarray;
begin
    Parameters.Clear;
    Parameters.Add( TPair<String,TValue>.Create('y',y) );
    Parameters.Add( TPair<String,TValue>.Create('axis',axis) );
    Parameters.Add( TPair<String,TValue>.Create('order',order) );

    Result := TNDArray.Create( InvokeStaticMethod(caller,'normalize',Parameters) )
end;

procedure TUtil.PlotModel(model: TBaseModel; to_file: string = 'model.png'; show_shapes: Boolean = false; show_layer_names: Boolean = true; rankdir: string = 'TB'; expand_nested: Boolean = false; dpi: Integer = 96) ;
begin
    Parameters.Clear;
    Parameters.Add( TPair<String,TValue>.Create('model',model) );
    Parameters.Add( TPair<String,TValue>.Create('to_file',to_file) );
    Parameters.Add( TPair<String,TValue>.Create('show_shapes',show_shapes) );
    Parameters.Add( TPair<String,TValue>.Create('show_layer_names',show_layer_names) );
    Parameters.Add( TPair<String,TValue>.Create('rankdir',rankdir) );
    Parameters.Add( TPair<String,TValue>.Create('expand_nested',expand_nested) );
    Parameters.Add( TPair<String,TValue>.Create('dpi',dpi) );

    InvokeStaticMethod(caller,'plot_model',Parameters)
end;

procedure TUtil.ConfigTensorFlowBackend(intra_op_parallelism_threads: Integer; inter_op_parallelism_threads: Integer; allow_soft_placement: Boolean; cpu_device_count: Integer;gpu_device_count: Integer );
var
 tf,conf,kb,config,session : TPythonObject;
 deviceCount          : TPyDict;
begin
    tf   := TPythonObject.Create(ImportModule('tensorflow'));
    conf := TPythonObject.Create(ImportModule('tensorflow.compat.v1'));
    kb   := TPythonObject.Create(ImportModule('tensorflow.compat.v1.keras.backend'));

    deviceCount := TPyDict.Create;
    deviceCount['CPU'] := TPyInt.Create( cpu_device_count );
    deviceCount['GPU'] := TPyInt.Create( gpu_device_count );

    Parameters.Clear;
    Parameters.Add( TPair<String,TValue>.Create('intra_op_parallelism_threads',intra_op_parallelism_threads) );
    Parameters.Add( TPair<String,TValue>.Create('inter_op_parallelism_threads',inter_op_parallelism_threads) );
    Parameters.Add( TPair<String,TValue>.Create('allow_soft_placement',allow_soft_placement) );
    Parameters.Add( TPair<String,TValue>.Create('device_count',deviceCount) );
    config := InvokeStaticMethod(conf,'ConfigProto',Parameters,False) ;

    Parameters.Clear;
    Parameters.Add( TPair<String,TValue>.Create('config',config) );
    session := InvokeStaticMethod(conf,'Session',Parameters,False);

    Parameters.Clear;
    Parameters.Add( TPair<String,TValue>.Create('session',session) );
    InvokeStaticMethod(kb,'set_session',Parameters)
end;

procedure TUtil.device( device_name: string );
var
 tf : TPythonObject;

begin
    tf   := TPythonObject.Create(ImportModule('tensorflow'));

    Parameters.Clear;
    Parameters.Add( TPair<String,TValue>.Create('device_name',device_name) );

    InvokeStaticMethod(tf,'device',Parameters)
end;

{ TBaseModel }

constructor TBaseModel.Create;
begin
    inherited Create;
end;

procedure TBaseModel.Compile(optimizer: TStringOrInstance; loss: string; metrics: TArray<string>;
                              loss_weights: TArray<Double>; sample_weight_mode: PChar; weighted_metrics: TArray<string>;
                              target_tensors: TArray<TNDarray>);
begin
    Parameters.Clear;

    Parameters.Add( TPair<String,TValue>.Create('optimizer',optimizer.PyObject));
    Parameters.Add( TPair<String,TValue>.Create('loss',loss));
    Parameters.Add( TPair<String,TValue>.Create('metrics',TValue.FromArray<string>(metrics)));
    Parameters.Add( TPair<String,TValue>.Create('loss_weights',TValue.FromArray<Double>(loss_weights)));
    Parameters.Add( TPair<String,TValue>.Create('sample_weight_mode',sample_weight_mode));
    Parameters.Add( TPair<String,TValue>.Create('weighted_metrics',TValue.FromArray<string>(weighted_metrics)));
    Parameters.Add( TPair<String,TValue>.Create('target_tensors',TValue.FromArray<TNDarray>(target_tensors)));

    g_MyPyEngine.Py_XDecRef(InvokeMethod('compile',Parameters).Handle)
end;

function TBaseModel.Evaluate(x, y: TNDarray; batch_size: PInteger; verbose: Integer; sample_weight: TNDarray;
                                steps: PInteger; callbacks: TArray<TCallback>): TArray<Double>;
begin
    Parameters.Clear;

    Parameters.Add( TPair<String,TValue>.Create('x', x));
    Parameters.Add( TPair<String,TValue>.Create('y',y));

    if batch_size <> nil  then
       Parameters.Add( TPair<String,TValue>.Create('batch_size',batch_size^));

    Parameters.Add( TPair<String,TValue>.Create('verbose',verbose));
    Parameters.Add( TPair<String,TValue>.Create('sample_weight',sample_weight));

    if steps <> nil  then
        Parameters.Add( TPair<String,TValue>.Create('steps',steps^));

    if callbacks <> nil  then
        Parameters.Add( TPair<String,TValue>.Create('callbacks',TValue.FromArray<TCallback>(callbacks)));

    Result := InvokeMethod('evaluate',Parameters).AsArrayofDouble;
end;

function TBaseModel.EvaluateGenerator(generator: TSequence; steps: PInteger; callbacks: TArray<TCallback>;
                                        max_queue_size, workers: Integer; use_multiprocessing: Boolean; verbose: Integer): TArray<Double>;
var
  pyresult : TPythonObject;
begin
    Parameters.Clear;

    Parameters.Add( TPair<String,TValue>.Create('generator',generator));

    if steps <> nil  then
      Parameters.Add( TPair<String,TValue>.Create('steps',steps^));

    if callbacks <> nil  then
        Parameters.Add( TPair<String,TValue>.Create('callbacks',TValue.FromArray<TCallback>(callbacks)));

    Parameters.Add( TPair<String,TValue>.Create('max_queue_size',max_queue_size));
    Parameters.Add( TPair<String,TValue>.Create('workers',workers));
    Parameters.Add( TPair<String,TValue>.Create('use_multiprocessing',use_multiprocessing));
    Parameters.Add( TPair<String,TValue>.Create('verbose',verbose));

    pyresult := InvokeMethod('evaluate_generator',Parameters);

    Result := [];
    if pyresult = nil then Exit(Result);

    if not g_MyPyEngine.PyIter_Check(pyresult.Handle) then
    begin
        Result := Result + [ pyresult.AsDouble ];
        Exit;
    end;

    result := pyresult.AsArrayofDouble;

    g_MyPyEngine.Py_XDecRef(pyresult.handle);

end;

function TBaseModel.Fit(x, y: TNDarray; batch_size: PInteger; epochs, verbose: Integer; callbacks: TArray<TCallback>;
                          validation_split: Double; validation_data: TArray<TNDarray>; shuffle: Boolean;
                          class_weight: TDictionary<Integer, Double>; sample_weight: TNDarray; initial_epoch: Integer; steps_per_epoch,
                          validation_steps: PInteger): THistory;
var
  py : TPythonObject;
begin
    Parameters.Clear;

    Parameters.Add( TPair<String,TValue>.Create('x',x));
    Parameters.Add( TPair<String,TValue>.Create('y',y));

    if batch_size <> nil then  Parameters.Add( TPair<String,TValue>.Create('batch_size',batch_size^))
    else                       Parameters.Add( TPair<String,TValue>.Create('batch_size', TPythonObject.None ));

    Parameters.Add( TPair<String,TValue>.Create('epochs',epochs));
    Parameters.Add( TPair<String,TValue>.Create('verbose',verbose));

    if callbacks <> nil  then  Parameters.Add( TPair<String,TValue>.Create('callbacks',TValue.FromArray<TCallback>(callbacks)))
    else                       Parameters.Add( TPair<String,TValue>.Create('callbacks', TPythonObject.None ));

    Parameters.Add( TPair<String,TValue>.Create('validation_split',validation_split));

    if (validation_data <> nil) then Parameters.Add( TPair<String,TValue>.Create('validation_data', TValue.FromArray<TNDarray>(validation_data)))
    else                             Parameters.Add( TPair<String,TValue>.Create('validation_data', TPythonObject.None ));

    Parameters.Add( TPair<String,TValue>.Create('shuffle',shuffle));

    if (class_weight <> nil) then Parameters.Add( TPair<String,TValue>.Create('class_weight',ToDict(class_weight)))
    else                          Parameters.Add( TPair<String,TValue>.Create('class_weight', TPythonObject.None ));

    Parameters.Add( TPair<String,TValue>.Create('sample_weight',sample_weight));
    Parameters.Add( TPair<String,TValue>.Create('initial_epoch',initial_epoch));

    if steps_per_epoch <> nil then Parameters.Add( TPair<String,TValue>.Create('steps_per_epoch',steps_per_epoch^))
    else                           Parameters.Add( TPair<String,TValue>.Create('steps_per_epoch', TPythonObject.None ));

    if validation_steps <> nil then Parameters.Add( TPair<String,TValue>.Create('validation_steps',validation_steps^))
    else                            Parameters.Add( TPair<String,TValue>.Create('validation_steps', TPythonObject.None ));

    py := InvokeMethod('fit', Parameters);

    Result := THistory.Create(py);


end;

function TBaseModel.Fit(x, y: TNDarray; batch_size: PInteger; epochs, verbose: Integer; validation_data: TArray<TNDarray>): THistory;
var
  py : TPythonObject;
begin
    Parameters.Clear;

    Parameters.Add( TPair<String,TValue>.Create('x',x));
    Parameters.Add( TPair<String,TValue>.Create('y',y));

    if batch_size <> nil then  Parameters.Add( TPair<String,TValue>.Create('batch_size',batch_size^))
    else                       Parameters.Add( TPair<String,TValue>.Create('batch_size', TPythonObject.None ));

    Parameters.Add( TPair<String,TValue>.Create('epochs',epochs));
    Parameters.Add( TPair<String,TValue>.Create('verbose',verbose));


    if (validation_data <> nil) then Parameters.Add( TPair<String,TValue>.Create('validation_data', TValue.FromArray<TNDarray>(validation_data)))
    else                             Parameters.Add( TPair<String,TValue>.Create('validation_data', TPythonObject.None ));

    py := InvokeMethod('fit', Parameters);

    Result := THistory.Create(py);
end;

function TBaseModel.FitGenerator(generator: TSequence; steps_per_epoch: PInteger; epochs, verbose: Integer;
                                  callbacks: TArray<TCallback>; validation_data: TSequence; validation_steps: PInteger; validation_freq: Integer;
                                  class_weight: TDictionary<Integer, Double>; max_queue_size, workers: Integer; use_multiprocessing, shuffle: Boolean;
                                  initial_epoch: Integer): THistory;
begin
    Parameters.Clear;

    Parameters.Add( TPair<String,TValue>.Create('generator',generator));
    if steps_per_epoch <> nil then
       Parameters.Add( TPair<String,TValue>.Create('steps_per_epoch',steps_per_epoch^));

    Parameters.Add( TPair<String,TValue>.Create('epochs',epochs));
    Parameters.Add( TPair<String,TValue>.Create('verbose',verbose));

    if callbacks <> nil  then
        Parameters.Add( TPair<String,TValue>.Create('callbacks',TValue.FromArray<TCallback>(callbacks)));

    if (validation_data <> nil) then
       Parameters.Add( TPair<String,TValue>.Create('validation_data',validation_data));

    if validation_steps <> nil then
       Parameters.Add( TPair<String,TValue>.Create('validation_steps',validation_steps^));
    //Parameters.Add( TPair<String,TValue>.Create('validation_freq',validation_freq));
    Parameters.Add( TPair<String,TValue>.Create('class_weight',class_weight));
    Parameters.Add( TPair<String,TValue>.Create('max_queue_size',max_queue_size));
    Parameters.Add( TPair<String,TValue>.Create('workers',workers));
    Parameters.Add( TPair<String,TValue>.Create('use_multiprocessing',use_multiprocessing));
    Parameters.Add( TPair<String,TValue>.Create('shuffle',shuffle));
    Parameters.Add( TPair<String,TValue>.Create('initial_epoch',initial_epoch));

    var py := InvokeMethod('fit_generator', Parameters);

    Result := THistory.Create(py);
end;

class function TBaseModel.LoadModel(filepath: string; custom_objects: TDictionary<string, string>; compile: Boolean): TBaseModel;
var
  model: TBaseModel;
  dict : TPyDict;
  item : TPair<string, string>;
  args : TList< TPair<String,TValue> > ;
begin
    model := TBaseModel.Create;

    if custom_objects <> nil then
    begin
        dict := TPyDict.Create;
        for item in custom_objects do
              dict[item.Key] := ToPython(Item.Value)  ;
    end;

    args := TList< TPair<String,TValue> >.Create;

    args.Add( TPair<String,TValue>.Create('filepath',filepath));

    if custom_objects <> nil then args.Add( TPair<String,TValue>.Create('custom_objects',ToDict(custom_objects)))
    else                          args.Add( TPair<String,TValue>.Create('custom_objects', TPythonObject.None ));

    model.PyInstance := GetKerasClassIstance('models');
    model.PyInstance := InvokeStaticMethod(model.PyInstance,'load_model',args) ;

    Result := model;

end;

class function TBaseModel.ModelFromJson(json_string: string): TBaseModel;
var
  model: TBaseModel;
  args : TList< TPair<String,TValue> > ;
begin
    model := TBaseModel.Create;

    args := TList< TPair<String,TValue> >.Create;

    args.Add( TPair<String,TValue>.Create('json_string',json_string));

    model.PyInstance := GetKerasClassIstance('models');
    model.PyInstance := InvokeStaticMethod(model.PyInstance,'model_from_json',args) ;

    Result := model;
end;

function TBaseModel.ModelFromYaml(Yaml_string: string): TBaseModel;
var
  model: TBaseModel;
begin
    model := TBaseModel.Create;

    Parameters.Clear;
    Parameters.Add( TPair<String,TValue>.Create('Yaml_string',Yaml_string));

    model.PyInstance := GetKerasClassIstance('models');
    model.PyInstance := InvokeStaticMethod(model.PyInstance,'model_from_yaml',Parameters) ;

    Result := model;
end;

function TBaseModel.GetWeights: TArray<TNDarray>;
var
  pyWeights   : TPythonObject;
  weights     : TArray<TNDarray>;
  i           : Integer;
  weightsArray: PPyObject;
begin
    Parameters.Clear;

    pyWeights := InvokeMethod('get_weights',Parameters) ;

    for i := 0 to g_MyPyEngine.PySequence_Length( pyWeights.Handle )-1 do
    begin
        weightsArray := g_MyPyEngine.PySequence_GetItem( pyWeights.Handle, i ) ;

        var n : TNDarray := TNumPy.npArray( TNDarray.Create(weightsArray) );
        weights := weights + [ n ];
    end;

    Result := weights;
end;

procedure TBaseModel.SetWeights(weights: TArray<TNDarray>);
var
 list : TPyList;
 i    : Integer;
begin
    list := TPyList.Create;

    for i := 0 to High(weights) do
      list.Append(weights[i]);

    Parameters.Clear;

    Parameters.Add( TPair<String,TValue>.Create('list', list));

    g_MyPyEngine.Py_XDecRef(InvokeMethod('set_weights',Parameters).Handle)
end;

procedure TBaseModel.SaveWeight(path: string);
begin
    Parameters.Clear;

    Parameters.Add( TPair<String,TValue>.Create('path', path));

    g_MyPyEngine.Py_XDecRef(InvokeMethod('save_weights',Parameters).Handle)
end;

procedure TBaseModel.LoadWeight(path: string);
begin
    Parameters.Clear;

    Parameters.Add( TPair<String,TValue>.Create('path', path));

    g_MyPyEngine.Py_XDecRef(InvokeMethod('load_weights',Parameters).Handle)
end;

procedure TBaseModel.Save(filepath: string; overwrite, include_optimizer: Boolean);
begin
    Parameters.Clear;

    Parameters.Add( TPair<String,TValue>.Create('filepath', filepath));
    Parameters.Add( TPair<String,TValue>.Create('overwrite', overwrite));
    Parameters.Add( TPair<String,TValue>.Create('include_optimizer', include_optimizer));

    g_MyPyEngine.Py_XDecRef(InvokeMethod('save',Parameters).Handle)
end;

procedure TBaseModel.Summary(line_length: PInteger; positions: TArray<Double>);
begin
    Parameters.Clear;

    if line_length <> nil then  Parameters.Add( TPair<String,TValue>.Create('line_length', line_length^))
    else                        Parameters.Add( TPair<String,TValue>.Create('line_length', TPythonObject.None ));

    if positions <> nil  then   Parameters.Add( TPair<String,TValue>.Create('positions', TValue.FromArray<Double>(positions)))
    else                        Parameters.Add( TPair<String,TValue>.Create('positions', TPythonObject.None ));

    g_MyPyEngine.Py_XDecRef(InvokeMethod('summary',Parameters).Handle );

end;

function TBaseModel.ToJson: string;
begin
    Parameters.Clear;

    Result  := InvokeMethod('to_json',Parameters).ToString;
end;

function TBaseModel.Predict(x: TArray<TNDarray>; batch_size: PInteger; verbose: Integer; steps: PInteger;
                                   callbacks: TArray<TCallback>): TNDarray;
var
  items  : TArray<PPyObject>;
  x_tuple: TPyTuple;
  n      : Integer;
begin
    Parameters.Clear;

    for n := 0  to Length(x) - 1 do
       items :=  items + [ x[n].Handle ];

    x_tuple := TPyTuple.Create(items);

    Parameters.Add( TPair<String,TValue>.Create('x',x_tuple));
    if batch_size <> nil then
        Parameters.Add( TPair<String,TValue>.Create('batch_size',batch_size^));
    Parameters.Add( TPair<String,TValue>.Create('verbose',verbose));
    if steps <> nil then
        Parameters.Add( TPair<String,TValue>.Create('steps',steps^));
    if callbacks <> nil  then
        Parameters.Add( TPair<String,TValue>.Create('callbacks',TValue.FromArray<TCallback>(callbacks)));

    Result := TNDarray.Create(InvokeMethod('predict', Parameters));
end;

function TBaseModel.Predict(x: TNDarray; batch_size: PInteger; verbose: Integer; steps: PInteger;
                                   callbacks: TArray<TCallback>): TNDarray;
begin
    Parameters.Clear;

    Parameters.Add( TPair<String,TValue>.Create('x',x));
    if batch_size <> nil then
        Parameters.Add( TPair<String,TValue>.Create('batch_size',batch_size^));
    Parameters.Add( TPair<String,TValue>.Create('verbose',verbose));
    if steps <> nil then
        Parameters.Add( TPair<String,TValue>.Create('steps',steps^));
    if callbacks <> nil  then
        Parameters.Add( TPair<String,TValue>.Create('callbacks',TValue.FromArray<TCallback>(callbacks)));

    Result := TNDarray.Create(InvokeMethod('predict', Parameters));
end;

function TBaseModel.PredictGenerator(generator: TSequence; steps: PInteger; callbacks: TArray<TCallback>;
                                        max_queue_size, workers: Integer; use_multiprocessing: Boolean; verbose: Integer): TNDarray;
var
  py : TPythonObject;
begin
    Parameters.Clear;

    Parameters.Add( TPair<String,TValue>.Create('generator',generator));
    if steps <> nil then
        Parameters.Add( TPair<String,TValue>.Create('steps',steps^));
    if callbacks <> nil  then
        Parameters.Add( TPair<String,TValue>.Create('callbacks',TValue.FromArray<TCallback>(callbacks)));
    Parameters.Add( TPair<String,TValue>.Create('max_queue_size',max_queue_size));
    Parameters.Add( TPair<String,TValue>.Create('workers',workers));
    Parameters.Add( TPair<String,TValue>.Create('use_multiprocessing',use_multiprocessing));
    Parameters.Add( TPair<String,TValue>.Create('verbose',verbose));

    py := InvokeMethod('predict_generator', Parameters);

    Result := TNDarray.Create(py);
end;

function TBaseModel.PredictOnBatch(x: TNDarray): TNDarray;
begin
    Parameters.Clear;

    Parameters.Add( TPair<String,TValue>.Create('x',x));

    Result := TNDarray.Create(InvokeMethod('predict_on_batch', Parameters));
end;

function TBaseModel.TrainOnBatch(x, y, sample_weight: TNDarray; class_weight: TDictionary<Integer, Double>): TArray<Double>;
var
  pyresult : TPythonObject;
begin
    Parameters.Clear;

    Parameters.Add( TPair<String,TValue>.Create('x',x));
    Parameters.Add( TPair<String,TValue>.Create('y',y));
    Parameters.Add( TPair<String,TValue>.Create('sample_weight',sample_weight));
    Parameters.Add( TPair<String,TValue>.Create('class_weight',class_weight));

    pyresult := InvokeMethod('train_on_batch', Parameters);

    Result := [];
    if pyresult = nil then Exit(Result);

    if not g_MyPyEngine.PyIter_Check(pyresult.Handle) then
    begin
        Result := Result + [ pyresult.AsDouble ];
        Exit;
    end;

    result := pyresult.AsArrayofDouble;
end;

function TBaseModel.TestOnBatch(x, y, sample_weight: TNDarray): TArray<Double>;
var
  pyresult : TPythonObject;
begin
    Parameters.Clear;

    Parameters.Add( TPair<String,TValue>.Create('x',x));
    Parameters.Add( TPair<String,TValue>.Create('y',y));
    Parameters.Add( TPair<String,TValue>.Create('sample_weight',sample_weight));

    pyresult := InvokeMethod('test_on_batch', Parameters);

    Result := [];
    if pyresult = nil then Exit(Result);

    if not g_MyPyEngine.PyIter_Check(pyresult.Handle) then
    begin
        Result := Result + [ pyresult.AsDouble ];
        Exit;
    end;

    result := pyresult.AsArrayofDouble;
    g_MyPyEngine.Py_XDecRef(pyresult.Handle);
end;

procedure TBaseModel.SaveOnnx(filePath: string);
begin
    Parameters.Clear;

    Parameters.Add( TPair<String,TValue>.Create('filePath',filePath));

    InvokeStaticMethod(hkeras2onnxMod,'convert_keras',Parameters) ;
    TFile.WriteAllText(filePath, hkeras2onnxMod.ToString);
end;

procedure TBaseModel.SaveTensorflowJSFormat(artifacts_dir: string; quantize: Boolean);
var
 htf : TPythonObject;
begin
    Parameters.Clear;

    htf :=  GetTFJSClassIstance('converters');

    if htf = nil then
    begin
        MessageBoxA(0,'tensorflowjs not installated!','Info',MB_OK);
        Exit;
    end;

    Parameters.Add( TPair<String,TValue>.Create('model',PyInstance));
    Parameters.Add( TPair<String,TValue>.Create('artifacts_dir',artifacts_dir));
    Parameters.Add( TPair<String,TValue>.Create('quantize',quantize));

    InvokeStaticMethod(htf,'save_keras_model',Parameters) ;

end;

{ TModel }

constructor TModel.Create;
begin
    inherited Create;
end;


constructor TModel.Create(py: TPythonObject);
begin
    Create;
    PyInstance :=  py;
end;

constructor TModel.Create(py: PPyObject);
begin
    Create;
    PyInstance := TPythonObject.Create( py );
end;

constructor TModel.Create(input: TArray<TBaseLayer>);
begin
    inherited Create;

    Parameters.Add( TPair<String,TValue>.Create('input', TValue.FromArray<TBaseLayer>(input)) );

    PyInstance := GetKerasClassIstance('models.Model');
    init();
end;

constructor TModel.Create(input, output: TArray<TBaseLayer>);
begin
    inherited Create;

    Parameters.Add( TPair<String,TValue>.Create('inputs', TValue.FromArray<TBaseLayer>(input)) );
    Parameters.Add( TPair<String,TValue>.Create('outputs', TValue.FromArray<TBaseLayer>(output)) );

    PyInstance := GetKerasClassIstance('models.Model');
    init();
end;

{ TSequential }

constructor TSequential.Create;
begin
     inherited Create;

     PyInstance := GetKerasClassIstance('models.Sequential');
     init;
end;

constructor TSequential.Create(py: PPyObject);
begin
    Create;
    PyInstance := TPythonObject.Create( py );
end;

constructor TSequential.Create(py: TPythonObject);
begin
    Create;
    PyInstance :=  py ;
end;

constructor TSequential.Create(input: TArray<TBaseLayer>);
var
  i : Integer;
begin
    for i := 0 to Length(input)-1 do
      Add( input[i]);
end;

procedure TSequential.Add(layer: TBaseLayer);
begin
    Parameters.Clear;

   Parameters.Add( TPair<String,TValue>.Create('layers', layer  ));

   g_MyPyEngine.Py_XDecRef ( InvokeMethod('add',Parameters).Handle ) ;
end;

end.
