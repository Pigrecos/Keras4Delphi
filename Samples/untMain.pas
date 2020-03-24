unit untMain;

interface

uses
  Winapi.Windows, Winapi.Messages, System.SysUtils, System.Variants, System.Classes, Vcl.Graphics,
  Vcl.Controls, Vcl.Forms, Vcl.Dialogs, Vcl.ExtCtrls, Vcl.StdCtrls, Vcl.ComCtrls,
  System.TypInfo, System.Rtti,PythonEngine, PythonGUIInputOutput,
  Python.Utils,

  Keras.Layers,
  Keras.Models,
  Keras,

  np.Base,
  np.Api,

  np.Models,
  NDArray.Api, VclTee.TeeGDIPlus, VCLTee.TeEngine, VCLTee.Series, VCLTee.TeeProcs, VCLTee.Chart, Vcl.Buttons;

type
  TArray2D<T> = array of TArray<T>;
  TfrmMain = class(TForm)
    pnlTop: TPanel;
    pnl1: TPanel;
    redtOutput: TRichEdit;
    PyIOCom: TPythonGUIInputOutput;
    btn1: TButton;
    img1: TImage;
    pnl2: TPanel;
    spl1: TSplitter;
    spl2: TSplitter;
    cht1: TChart;
    srsTraining_Loss: TLineSeries;
    srsValidation_loss: TLineSeries;
    procedure FormShow(Sender: TObject);
    procedure btn1Click(Sender: TObject);
    procedure redtOutputChange(Sender: TObject);
  private
    procedure TestVarios;
    procedure esempio_XOR;
    procedure MergeExample;
    procedure ImplementCallback;
    procedure NumPyTest;
    procedure Test1;
    procedure MNIST_CNN;
    procedure SentimentClassification;
    procedure SentimentClassificationLSTM;
    procedure Predict(text: string);
    procedure TextGen;
    function LoadTxt(fileName: string; lenSeq:Integer; step: Integer; var rawTxt: TArray<AnsiChar>; var DataX: TArray2D<Integer>; var DataY: TArray<Integer>):Integer;
    procedure TextGen_predict(model : TSequential; seqLen, Step: Integer);
    { Private declarations }
  public
    { Public declarations }
  end;

var
  frmMain: TfrmMain;
  vNumpy : TNumPy;


implementation
     uses System.Generics.Collections, System.Diagnostics, System.IOUtils, Jpeg,MethodCallBack,System.Math,
          np.Utils,

          Keras.PreProcessing;

{$R *.dfm}

procedure TfrmMain.TestVarios;
var
  shape,shape1 : Tnp_Shape;
  TestArray    : TNDArray<int64>;
  TestArray1   : TNDArray;
  aN1,aN2      : TNDArray;
  s1           : string;
  i            : Integer;
begin

    TestArray  := vNumpy.npArray<Int64>(  [1,2,3]) ;
    s1 := TestArray.repr ;
    redtOutput.Lines.Add(s1) ;
    redtOutput.Lines.Add('==========');

    TestArray := vNumpy.npArray<Int64>(  [ [1,2,3], [1,2,3] ]) ;
    s1 := TestArray.repr ;
    redtOutput.Lines.Add(s1) ;
    redtOutput.Lines.Add('==========');

    TestArray := vNumpy.npArray<Int64>([ [ [2,4,5], [7,8,9] ],
                                         [ [6,3,2], [8,2,1] ]
                                       ]  ) ;

    TestArray.data;
    TestArray.ndim;
    TestArray.itemsize;
    TestArray.len;
    TestArray.item<int64>([4]);


    var a    : TArray2D<Integer> :=   [ [1,2,3], [4,5,6] ] ;
    var given: TNDArray          :=  vNumpy.empty_like<Integer>(a);
    var sh   : Tnp_Shape         := given.shape;
    var dt   : TDtype            := given.dtype;
    redtOutput.Lines.Add(dt.ToString);

    s1 := TestArray.repr ;
    redtOutput.Lines.Add(s1) ;
    redtOutput.Lines.Add('==========');

    TestArray1 := vNumpy.npArray(['uno','due','tre','quattro','cinque']  ) ;
    s1 := TestArray1.repr ;
    redtOutput.Lines.Add(s1) ;
    redtOutput.Lines.Add('==========');

    shape := Tnp_Shape.Create([1,2,3]);
    shape1:= Tnp_Shape.Create([7,4,5]);

    aN1  := TNDArray.Create(TNumPy.ToPython( TValue.FromShape(shape1)));
    aN2  := TNDArray.Create(TNumPy.ToPython( TValue.FromShape(shape)));
    TestArray1 := vNumpy.npArray([aN1,aN2]  ) ;
    s1 := TestArray1.repr ;
    redtOutput.Lines.Add(s1) ;
    redtOutput.Lines.Add('==========');


    TestArray1 := vNumpy.npArray<Integer>([24]);
    i :=vNumpy.asscalar<Integer>(TestArray1);
    redtOutput.Lines.Add( IntToStr(i) )

end;

function log(epochidx: Integer): Double;
begin
    Result := 0.0;
    frmMain.redtOutput.Lines.Add('test callback :' + IntToStr(epochidx));
end;

function pylog(self, args : PPyObject): PPyObject;cdecl;
var
  epochidx: Integer ;
  d       : Double;
begin
    Result := nil;
    if g_MyPyEngine.PyArg_ParseTuple( args, 'if:logFunction',@epochidx,@d) <> 0 then
    begin
        log(epochidx) ;
        d := 7.0;
        Result :=   g_MyPyEngine.VariantAsPyObject(d) ;
    end;
end;

procedure TfrmMain.btn1Click(Sender: TObject);
begin
    //============== Esempi ======================//

    // ====NumPy basic test
    (*NumPyTest;

    // ====keras test
    Test1;
    //
    esempio_XOR;
    //
    MergeExample ;
    //
    ImplementCallback;
    //
    MNIST_CNN ;
    //
    SentimentClassification;
    Predict('I hate you');
    Predict('I care about you');
    //
    SentimentClassificationLSTM;  *)
    //
    TextGen ;
    var filemodel : string      :='TextGen.h5';
    var model     : TSequential := TSequential(TSequential.LoadModel(filemodel));
    TextGen_predict(model,40,3);
    //
end;

procedure TfrmMain.NumPyTest;
var
  stopwatch         : TStopwatch ;
  N, D_in, HH, D_out : Integer;
  x,y,w1,w2         : TNDArray;
  learning_rate,
  loss              : Double;
  t : Integer;
begin
    TestVarios;

    N := 64; D_in := 1000; HH := 100; D_out := 10;

    // Create random input and output data
    redtOutput.Lines.Add(' creating random data');
    x  := vNumpy.randn([N, D_in]);
    y  := vNumpy.randn([N, D_out]);

    y.item<double>([1,2]);

    redtOutput.Lines.Add(' learning');
    stopwatch :=  TStopwatch.StartNew;
    // Randomly initialize weights
    w1 := vNumpy.randn([D_in, HH]);
    w2 := vNumpy.randn([HH, D_out]);

    learning_rate := 1.0E-06;

    var h           : TNDArray;
    var h_relu      : TNDArray;
    var y_pred      : TNDArray;
    var grad_y_pred : TNDArray;
    var grad_w2     : TNDArray;
    var grad_h_relu : TNDArray;
    var grad_h      : TNDArray;
    var grad_w1     : TNDArray;
    for t := 0 to 500 do
    begin
        // Forward pass: compute predicted y
        h      := x.dot(w1);
        h_relu := vNumpy.maximum(h, vNumpy.asarray(0));
        y_pred := h_relu.dot(w2);

        // Compute and print loss
        loss :=  vNumpy.asscalar<Double>( vNumpy.square(TNDArray.opSub(y_pred, y)).sum);
        if (t mod 20) = 0 then
           redtOutput.Lines.Add( Format('step: %d loss: %e ',[t,loss]));

        // Backprop to compute gradients of w1 and w2 with respect to loss
        grad_y_pred := TNDArray.opMul(2.0, (TNDArray.opSub(y_pred, y)));
        grad_w2     := h_relu.T.dot(grad_y_pred);
        grad_h_relu := grad_y_pred.dot(w2.T);
        grad_h      := grad_h_relu.copy();
        TPythonObject(grad_h)[TNDArray.opLess(h, 0)]   := vNumpy.asarray(0);
        grad_w1     := x.T.dot(grad_h);

        // Update weights
        w1.opISub( TNDArray.opMul(learning_rate , grad_w1)); // inplace substraction is faster than -=
        w2.opISub( TNDArray.opMul(learning_rate , grad_w2));
    end;
    stopwatch.Stop;
    stopwatch.ElapsedMilliseconds;
    redtOutput.Lines.Add( Format('step: 500, final loss: %e , elapsed time: %s',[loss,formatdatetime('hh:nn:ss.zzz', stopwatch.ElapsedMilliseconds)]));

end;

procedure TfrmMain.redtOutputChange(Sender: TObject);
begin
    SendMessage(redtOutput.handle, WM_VSCROLL, SB_BOTTOM, 0);
end;

procedure TfrmMain.Test1;
var
 res      : tarray<tndarray>;

begin


    res := TCifar100.load_data;
    var x_train, y_train ,x_test, y_test : TNDArray;

    x_train := res[0];
    y_train := res[1];
    x_test  := res[2];
    y_test  := res[3];

    redtOutput.Lines.Add('x_train shape: ' + x_train.shape.ToString);
    redtOutput.Lines.Add( IntToStr(x_train.shape[0]) + ' train samples');
    redtOutput.Lines.Add( IntToStr(x_test.shape[0]) + '  test  samples');

    redtOutput.Lines.Add('y_train shape: ' + y_train.shape.ToString);
    redtOutput.Lines.Add( IntToStr(y_train.shape[0]) + ' train samples');
    redtOutput.Lines.Add( IntToStr(y_test.shape[0]) + '  test  samples');

    TEarlyStopping.Create ('val_loss', 3)  ;

    TLearningRateScheduler.Create (pylog)  ;
end;

procedure TfrmMain.esempio_XOR;
begin
    //Load train data
    var x : TNDarray := TNumPy.npArray<Double>( [ [ 0, 0 ], [ 0, 1 ], [ 1, 0 ], [ 1, 1 ] ] );
    var y : TNDarray := TNumPy.npArray<Double>( [ 0, 1, 1, 0 ] );

    //Build functional model
    var input  : TKInput := TKInput.Create(tnp_shape.Create([2]));
    var hidden1: TBaseLayer  := TDense.Create(32, 'relu').&Set([input]);
    var hidden2: TBaseLayer  := TDense.Create(64, 'relu').&Set([hidden1]);
    var output : TBaseLayer  := TDense.Create(1,  'sigmoid').&Set([hidden2]);

    var model : TModel := TModel.Create ( [ input ] , [ output ]);
    model.Summary;

    //Compile and train
    model.Compile(TAdam.Create , 'binary_crossentropy',['accuracy']);

    var batch_size : Integer := 2;
    var history: THistory := model.Fit(x, y, @batch_size, 10,1);

    srsTraining_Loss.Clear;
    srsTraining_Loss.AddArray(history.HistoryLogs['loss']) ;
    srsValidation_loss.Clear;
    srsValidation_loss.AddArray(history.HistoryLogs['accuracy']);

    history.HistoryLogs;

    //Save model and weights
    var json : string := model.ToJson;
    TFile.WriteAllText('model.json', json);
    model.SaveWeight('model.h5');

    //Load model and weight
    var loaded_model : TBaseModel := TSequential.ModelFromJson(TFile.ReadAllText('model.json'));
    loaded_model.LoadWeight('model.h5');
end;

procedure TfrmMain.MergeExample;
begin
    var input  : TKInput     := TKInput.Create(tnp_shape.Create([32, 32]));
    var a      : TBaseLayer  := TDense.Create(32, 'sigmoid').&Set([input]);
    var output : TBaseLayer  := TDense.Create(1, 'sigmoid').&Set([a]);

    TModel.Create ( [ input ] , [ output ]);

    //Load train data
    var x : TNDarray := TNumPy.npArray<Double>( [ [ 0, 0 ], [ 0, 1 ], [ 1, 0 ], [ 1, 1 ] ] );
    redtOutput.Lines.Add('x :' + x.shape.ToString);
    TNumPy.npArray<Double>( [ 0, 1, 1, 0 ] );

    var input1  : TKInput    := TKInput.Create(tnp_shape.Create([32, 32,3]));
    var conv1   : TBaseLayer := TConv2D.Create(32, [4, 4], 'relu').&Set([input1]);
    var pool1   : TBaseLayer := TMaxPooling2D.Create([2, 2]).&Set([conv1]);
    var flatten1: TBaseLayer := TFlatten.Create.&Set([pool1]);

    var input2  : TKInput    := TKInput.Create(tnp_shape.Create([32, 32,3]));
    var conv2   : TBaseLayer := TConv2D.Create(16, [8, 8], 'relu').&Set([input2]);
    var pool2   : TBaseLayer := TMaxPooling2D.Create([2, 2]).&Set([conv2]);
    var flatten2: TBaseLayer := TFlatten.Create.&Set([pool2]);

    TConcatenate.Create([flatten1, flatten2]);

end;

procedure TfrmMain.ImplementCallback;
begin

    //Load train data
    var x : TNDarray := TNumPy.npArray<Double>( [ [ 0, 0 ], [ 0, 1 ], [ 1, 0 ], [ 1, 1 ] ] );
    var y : TNDarray := TNumPy.npArray<Double>( [ 0, 1, 1, 0 ] );

    //Build sequential model
    var model : TSequential := TSequential.Create;

    var inp : Tnp_shape := Tnp_shape.Create([2]);
    model.Add( TDense.Create(32, 'relu', @inp) );
    model.Add( TDense.Create(64, 'relu'));
    model.Add( TDense.Create(1, 'sigmoid'));

    var callback : keras.TCallback := TLearningRateScheduler.Create (pylog) ;

    var lossHistory : keras.TCallback := keras.TCallback.Custom('LossHistory', 'LossHistory.py');


    //Compile and train
    model.Compile(TAdam.Create, 'binary_crossentropy', [ 'accuracy' ]);
    var batch_size: Integer  := 2;
    model.Fit(x, y, @batch_size, 10, 1, [ callback,lossHistory ]);

    var customLosses: TArray<Double> := lossHistory.GetDoubleArray('losses');

end;

procedure TfrmMain.MNIST_CNN;
var
  res      : TArray<TNDArray>;
begin
    var batch_size : Integer := 128; //Training batch size
    var num_classes: Integer := 10;  //No. of classes
    var epochs     : Integer := 12;  //No. of epoches we will train

    // input image dimensions
    var img_rows: Integer := 28;
    var img_cols: Integer := 28;

    // Declare the input shape for the network
    var input_shape : Tnp_shape := default(Tnp_shape);

    // Load the MNIST dataset into Numpy array
    res := TMNIST.load_data;
    var x_train, y_train ,x_test, y_test : TNDArray;

    x_train := res[0];
    y_train := res[1];
    x_test  := res[2];
    y_test  := res[3];
    redtOutput.Lines.Add('y_train shape: ' + y_train.shape.ToString);
     redtOutput.Repaint;
    //Check if its channel fist or last and rearrange the dataset accordingly
    var K: TBackend := TBackend.Create;
    if(K.ImageDataFormat = 'channels_first') then
    begin
        x_train := x_train.reshape([x_train.shape[0], 1, img_rows, img_cols]);
        x_test  := x_test.reshape ([x_test.shape[0] , 1, img_rows, img_cols]);
        input_shape := Tnp_shape.Create([1, img_rows, img_cols]);
    end else
    begin
        x_train := x_train.reshape([x_train.shape[0], img_rows, img_cols, 1]);
        x_test  := x_test.reshape ([x_test.shape[0] , img_rows, img_cols, 1]);
        input_shape := Tnp_shape.Create([img_rows, img_cols, 1]);
    end;

    //Normalize the input data
    x_train := x_train.astype(vNumpy.float32_);
    x_test  := x_test.astype(vNumpy.float32_);
    x_train := TNDArray.opDiv(x_train, 255);
    x_test  := TNDArray.opDiv(x_test, 255);

    redtOutput.Lines.Add('x_train shape: ' + x_train.shape.ToString);
    redtOutput.Lines.Add('y_train shape: ' + y_train.shape.ToString);
    redtOutput.Lines.Add( IntToStr(x_train.shape[0]) + ' train samples');
    redtOutput.Lines.Add( IntToStr(x_test.shape[0]) + '  test  samples');
    redtOutput.Repaint;

    // Convert class vectors to binary class matrices
    var Util : TUtil := TUtil.Create;
    y_train := Util.ToCategorical(y_train, @num_classes);
    y_test  := Util.ToCategorical(y_test, @num_classes);
    redtOutput.Lines.Add('y_train shape: ' + y_train.shape.ToString)   ;
    redtOutput.Repaint;

    // Build CNN model
    var model : TSequential := TSequential.Create;

    model.Add( TConv2D.Create(32, [3, 3],'relu', @input_shape) );
    model.Add( TConv2D.Create(64, [3, 3],'relu'));
    model.Add( TMaxPooling2D.Create([2, 2]));
    model.Add( TDropout.Create(0.25));
    model.Add( TFlatten.Create);
    model.Add( TDense.Create(128, 'relu'));
    model.Add( TDropout.Create(0.5));
    model.Add( TDense.Create(num_classes, 'softmax'));
    model.Summary;

    //Compile with loss, metrics and optimizer
    model.Compile(TAdadelta.Create, 'categorical_crossentropy', [ 'accuracy' ]);

    //Train the model
    var history: THistory := model.Fit(x_train, y_train, @batch_size, epochs, 1,nil,0,[ x_test, y_test ]);

    srsTraining_Loss.Clear;
    srsTraining_Loss.AddArray(history.HistoryLogs['loss']) ;
    srsValidation_loss.Clear;
    srsValidation_loss.AddArray(history.HistoryLogs['accuracy']);

    //Score the model for performance
    var score : TArray<Double> := model.Evaluate(x_test, y_test, nil, 0);

    redtOutput.Lines.Add('Test loss: '   + FloatToStr(score[0]));
    redtOutput.Lines.Add('Test accuracy:'+ FloatToStr(score[1]));

    // Save the model to HDF5 format which can be loaded later or ported to other application
    model.Save('model.h5');
    // Save it to Tensorflow JS format and we will test it in browser.
    model.SaveTensorflowJSFormat('./');
end;

//https://keras.io/examples/imdb_lstm/
procedure TfrmMain.SentimentClassificationLSTM;
var
  res      : TArray<TNDArray>;
begin
    var max_features: Integer := 20000;
    // cut texts after this number of words (among top max_features most common words)
    var maxlen     : Integer := 80;
    var batch_size : Integer := 32;
    var EpochNum   : Integer := 15;

    redtOutput.Lines.Add('Loading data...');
    res := TIMDB.load_data(@max_features);
    var x_train, y_train ,x_test, y_test : TNDArray;

    x_train := res[0];
    y_train := res[1];
    x_test  := res[2];
    y_test  := res[3];

    redtOutput.Lines.Add('train sequences: ' + x_train.shape.ToString);
    redtOutput.Lines.Add('test sequences: '  + x_test.shape.ToString);

    redtOutput.Lines.Add('Pad sequences (samples x time)');
    var tseq : TSequenceUtil := TSequenceUtil.Create;
    x_train := tseq.PadSequences(x_train, @maxlen);
    x_test  := tseq.PadSequences(x_test,  @maxlen);
    redtOutput.Lines.Add('x_train shape: ' + x_train.shape.ToString);
    redtOutput.Lines.Add('x_test shape: '  + x_test.shape.ToString);

    redtOutput.Lines.Add('Build model...');
    var model : TSequential := TSequential.Create;
    model.Add( TEmbedding.Create(max_features, 128));
    model.Add( TLSTM.Create(128, 0.2, 0.2));
    model.Add( TDense.Create(1, 'sigmoid'));

    //try using different optimizers and different optimizer configs
    model.Compile('rmsprop', 'binary_crossentropy', [ 'accuracy' ]);
    model.Summary;

    redtOutput.Lines.Add('Train...');
    var history: THistory := model.Fit(x_train, y_train, @batch_size, EpochNum, 1,[ x_test, y_test ]);

    srsTraining_Loss.Clear;
    srsTraining_Loss.AddArray(history.HistoryLogs['loss']) ;
    srsValidation_loss.Clear;
    srsValidation_loss.AddArray(history.HistoryLogs['accuracy']);

    //Score the model for performance
    var score : TArray<Double> := model.Evaluate(x_test, y_test, @batch_size);

    redtOutput.Lines.Add('Test score: '   + FloatToStr(score[0]));
    redtOutput.Lines.Add('Test accuracy:'+ FloatToStr(score[1]));

    // Save the model to HDF5 format which can be loaded later or ported to other application
    model.Save('model.h5');

end;

procedure TfrmMain.SentimentClassification;
var
  res      : TArray<TNDArray>;

begin
    // Embedding
    var max_features  := 20000;
    var maxlen        := 500;
    var embedding_size:= 32;

    // Convolution
    var filters      := 32;
    var kernel_size  := 3;
    var pool_size    := 2;

    // Load the dataset but only keep the top n words, zero the rest
    res := TIMDB.load_data(@max_features);

    var x_train, y_train ,x_test, y_test: TNDArray;
    x_train := res[0];
    y_train := res[1];
    x_test  := res[2];
    y_test  := res[3];
    redtOutput.Lines.Add('x_train shape: ' + x_train.shape.ToString);
    redtOutput.Lines.Add('x_test shape: '  + x_test.shape.ToString);

    var tseq : TSequenceUtil := TSequenceUtil.Create;
    x_train := tseq.PadSequences(x_train, @maxlen);
    x_test  := tseq.PadSequences(x_test,  @maxlen);
    redtOutput.Lines.Add('x_train shape: ' + x_train.shape.ToString);
    redtOutput.Lines.Add('x_test shape: '  + x_test.shape.ToString);

    // Create model
    var model : TSequential := TSequential.Create;

    model.Add( TEmbedding.Create(max_features, embedding_size, @maxlen));
    model.Add( TConv1D.Create(filters, kernel_size, 'same', 'relu',nil));
    model.Add( TMaxPooling1D.Create(pool_size));
    model.Add( TFlatten.Create);
    model.Add( TDense.Create(250, 'relu'));
    model.Add( TDense.Create(1, 'sigmoid'));

    //Compile with loss, metrics and optimizer
    model.Compile('sgd', 'categorical_crossentropy', [ 'accuracy' ]);
    model.Summary;

    //Train the model
    var history : THistory := model.Fit(x_train, y_train, @embedding_size, 10, 2,[ x_test, y_test ]);

    srsTraining_Loss.Clear;
    srsTraining_Loss.AddArray(history.HistoryLogs['loss']) ;
    srsValidation_loss.Clear;
    srsValidation_loss.AddArray(history.HistoryLogs['accuracy']);

    //Score the model for performance
    var score : TArray<Double> := model.Evaluate(x_test, y_test, nil, 0);

    redtOutput.Lines.Add('Test loss: '   + FloatToStr(score[0]));
    redtOutput.Lines.Add('Test accuracy:'+ FloatToStr(score[1]*100));

    // Save the model to HDF5 format which can be loaded later or ported to other application
    model.Save('model.h5');
    // Save it to Tensorflow JS format and we will test it in browser.
    model.SaveTensorflowJSFormat('./');
end;

procedure TfrmMain.Predict(text: string);
var
  model   : TBaseModel;
  indexes : TDictionary<string, Integer>;
  words   : TArray<string>;
  w,res   : string;
  tokens  : TArray<Double>;
  TextUtil: TTextUtil;
begin
    model := TSequential.LoadModel('model.h5') ;

    indexes := TIMDB.GetWordIndex;

    TextUtil := TTextUtil.Create;
    Words    := TextUtil.TextToWordSequence(text);
    for w in words do
     tokens := tokens + [ indexes[w] ];

    var x : TNDArray := TNumPy.npArray<Double>(tokens);
    x := x.reshape([1,x.shape[0]]);
    var tseq : TSequenceUtil := TSequenceUtil.Create;
    var maxlen : Integer := 500;
    x := tseq.PadSequences(x, @maxlen);
    var y : TNDArray := model.Predict(x);

    var binary := Round(y.asscalar<double>);

    if binary = 0 then res := 'Negative'
    else               res := 'Positive' ;

    redtOutput.Lines.Add(Format('Sentiment for "{%s} : {%s}"',[ text, res]));

end;

procedure OnEpochEnd(epochidx: Integer;log:string);
begin
    frmMain.redtOutput.Lines.Add('test callback :' + IntToStr(epochidx));
end;

function pyOnEpochEnd(self, args : PPyObject): PPyObject;cdecl;  far;
var
  epochidx,i: Integer ;
  dLogs     : PPyObject;

begin
    Result := nil;
    if g_MyPyEngine.PyArg_ParseTuple( args, 'iO:OnEpochEnd', @epochidx,@dLogs  ) <> 0 then
    begin
        OnEpochEnd(epochidx,'') ;

        Result :=   g_MyPyEngine.VariantAsPyObject(epochidx) ;
    end;
end;

function TfrmMain.LoadTxt(fileName: string; lenSeq:Integer; step: Integer; var rawTxt: TArray<AnsiChar>; var DataX: TArray2D<Integer>; var DataY: TArray<Integer>):Integer;
var
  i,n,n_chars,
  seq_length    : Integer;
  s             : AnsiString;
  c             : AnsiChar;
  ss            : TStringStream;
  seq_in,
  seq_out       : TArray<AnsiChar>;
  char_to_int   : TDictionary<AnsiChar,Integer>;

begin
    Result := 0;

    ss := TStringStream.Create;
    try
      ss.LoadFromFile(fileName);
      SetLength(rawTxt,ss.Size);

      ss.Position := 0;
      i := ss.Read(rawTxt[0],ss.Size) ;

      if i <1 then Exit;

      SetString(s,PAnsiChar(@rawTxt[0]), Length(rawTxt));
      s := AnsiLowerCase(s);

      ZeroMemory(@rawTxt[0],Length(rawTxt));
      CopyMemory(@rawTxt[0],@s[1],Length(s));
      s := '';

      char_to_int := TDictionary<AnsiChar,Integer>.Create;
      try
        n := 0;
        for i := 0 to Length(rawTxt)-1 do
        begin
            if not char_to_int.ContainsKey( rawTxt[i] ) then
            begin
                char_to_int.Add(rawTxt[i],n);
                inc(n);
            end;
        end;
        Result :=  char_to_int.Count;

        n_chars := Length(rawTxt);
        // # prepare the dataset of input to output pairs encoded as integers
        seq_length := lenSeq;
        SetLength(seq_in,seq_length) ;
        SetLength(seq_out,1);
        i := 0;
        repeat
           ZeroMemory(seq_in,Length(seq_in));
           ZeroMemory(seq_out,Length(seq_out));
           TArray.Copy<AnsiChar>(rawTxt,seq_in,i,0, seq_length);
           TArray.Copy<AnsiChar>(rawTxt,seq_out,i+seq_length,0, 1);

           SetLength(dataX,Length(dataX)+1);
           for c in seq_in do
             dataX[ High(dataX) ] := dataX[ High(dataX) ] + [ char_to_int[c] ] ;

           dataY := dataY + [ char_to_int[seq_out[0]] ]  ;

           inc(i,Step);
        until i >= ((n_chars - seq_length)-1);

      finally
        char_to_int.Free;
      end;

    finally
      ss.Free
    end;
end;

// https://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/
// text file - http://www.gutenberg.org/ebooks/28371
procedure TfrmMain.TextGen;
var
  n_chars,i,t,
  n_vocab,
  seq_length,
  n_patterns,
  step         : Integer;
  raw_text     : TArray<AnsiChar>;
  dataX        : TArray2D<Integer>;
  dataY,
  sentence     : TArray<Integer>;
  item         : TArray<Integer>;
  X,Y          : TNDArray;
  input_shape  : Tnp_Shape;
  checkpoint,
  early_stop,
  logEpoch     : keras.TCallback ;
  batch_size   : Integer;
  epochs       : Integer;
begin
    seq_length := 40;
    batch_size := 64;
    epochs     := 20;
    step       := 3;

    n_vocab := LoadTxt('Alice.txt', seq_length, step,raw_text, dataX,dataY);
    n_chars    := Length(raw_text);
    n_patterns := Length(dataX);

    redtOutput.Lines.Add('Total Characters: '   + IntToStr(n_chars));
    redtOutput.Lines.Add('Total Vocab: '        + IntToStr(n_vocab));
    redtOutput.Lines.Add('Total Patterns: '     + IntToStr(n_patterns));

    //# reshape X to be [samples, time steps, features]
    X := TNumPy.zeros(Tnp_Shape.Create([n_patterns,seq_length,n_vocab]), vNumpy.int32_ ) ;
    Y := TNumPy.zeros(Tnp_Shape.Create([n_patterns,n_vocab]), vNumpy.int32_) ;

    //# one hot encode the input/output variable
    for i := 0 to n_patterns -1 do
    begin
        sentence := dataX[i];
        for t := 0 to High(sentence) do
        begin
            item := [i,t, sentence[t] ];
            X[ item ] := TNumPy.asarray(1);
        end;
        item  := [i, dataY[i]];
        Y[ item ] := TNumPy.asarray(1);
    end;

    input_shape := tnp_shape.Create([X.shape[1], X.shape[2]]);
    //# define the LSTM model
    var model : TSequential := TSequential.Create;
    var filepath : string :='weights-improvement-{epoch:02d}-{loss:.4f}.hdf5';
    model.Add( TBidirectional.Create( TGRU.Create(256),@input_shape ));
    model.Add( TDropout.Create(0.2));
    model.Add( TDense.Create(y.shape[1]));
    model.Add( TActivation.Create('softmax') );

    model.Compile('adam' ,'categorical_crossentropy', [ 'accuracy' ]);
    model.Summary;

    checkpoint := TModelCheckpoint.Create(filepath,'loss',1,True,False,'min');
    early_stop := TEarlyStopping.Create('val_acc',0,20);
    logEpoch   := TLambdaCallback.Create(pyOnEpochEnd);

    model.Fit(X,Y,@batch_size,epochs,1,[checkpoint,early_stop,logEpoch]) ;

    model.Save('TextGen.h5');

end;

procedure TfrmMain.TextGen_predict(model : TSequential; seqLen, Step: Integer);
var
  n,i,start,
  n_vocab,
  maxlen,t     : Integer;
  raw_text     : TArray<AnsiChar>;
  char_to_int  : TDictionary<AnsiChar,Integer>;
  int_to_char  : TDictionary<Integer,AnsiChar>;
  dataX        : TArray2D<Integer>;
  dataY        : TArray<Integer>;
  pattern,item : TArray<Integer>;
  s            : AnsiString;
  x,prediction,
  tmp          : TNDArray;
  index        : Integer;
begin
    tmp := nil;
    n_vocab := LoadTxt('Alice.txt', seqLen,Step,raw_text, dataX,dataY);
    maxlen  := Length (dataX[0]);

    char_to_int := TDictionary<AnsiChar,Integer>.Create;
    int_to_char := TDictionary<Integer,AnsiChar>.Create;
    try
      n := 0;
      for i := 0 to Length(raw_text)-1 do
      begin
          if not char_to_int.ContainsKey( raw_text[i] ) then
          begin
              char_to_int.Add(raw_text[i],n);
              int_to_char.Add(n,raw_text[i]);
              inc(n);
          end;
      end;
      //# pick a random seed
      start   := Random(Length(dataX)-1);
      pattern := dataX[start];

      s:= '';
      for i := 0 to High(pattern) do
        s := s + int_to_char[ pattern[i] ];

      redtOutput.Lines.Add('Seed: ');
      redtOutput.Lines.Add(s);
      redtOutput.Repaint;

      //# generate characters
      for i := 0 to 1000 do
      begin
          X := TNumPy.zeros(Tnp_Shape.Create([1,maxlen,n_vocab]), vNumpy.bool_ ) ;
          //# one hot encode the input/output variable
          for t := 0 to High(pattern) do
          begin
              item := [0,t, pattern[t]];
              X[ item ] := TNumPy.asarray(1);
          end;

          prediction := model.Predict(X,nil,0);
          index := vNumpy.argmax(prediction,nil,tmp).asscalar<integer>;
          pattern := pattern + [ index ];
          Delete(pattern,0,1);

          s := s + int_to_char[ index ];
      end;
      redtOutput.Lines.Add('Output: ');
      redtOutput.Lines.Add(s);
      redtOutput.Refresh;
    finally
      char_to_int.Free;
      int_to_char.Free;
    end;
end;

procedure TfrmMain.FormShow(Sender: TObject);
var
   jpg: TJpegImage;

begin
    // inizialige python engine with gui input/output
    InitGlobal(PyIOCom);

    vNumpy := TNumPy.Init(True);

    jpg := TJpegImage.Create;
    jpg.LoadFromFile('nn.jpg');
    img1.Picture.Assign(jpg);
end;

end.
