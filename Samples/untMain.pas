unit untMain;

interface

uses
  Winapi.Windows, Winapi.Messages, System.SysUtils, System.Variants, System.Classes, Vcl.Graphics,
  Vcl.Controls, Vcl.Forms, Vcl.Dialogs, Vcl.ExtCtrls, Vcl.StdCtrls, Vcl.ComCtrls,
  System.TypInfo, System.Rtti,PythonEngine, PythonGUIInputOutput,
  Python.Utils,

  np.Base,
  np.Api,

  Models,
  NDArray.Api;

type
  TArray2D<T> = array of TArray<T>;
  TfrmMain = class(TForm)
    pnlTop: TPanel;
    pnl1: TPanel;
    redtOutput: TRichEdit;
    splBottom: TSplitter;
    PyIOCom: TPythonGUIInputOutput;
    btn1: TButton;
    img1: TImage;
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
    { Private declarations }
  public
    { Public declarations }
  end;

var
  frmMain: TfrmMain;
  vNumpy : TNumPy;

implementation
     uses System.Generics.Collections, System.Diagnostics, System.IOUtils, Jpeg,MethodCallBack,
          utils,
          Keras,
          Keras.Layers,
          Keras.Models;

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

    var p: PPyObject := TestArray.data;
    var g: Integer   := TestArray.ndim;
    var gg : Integer := TestArray.itemsize;
    var gz : Integer := TestArray.len;
    var ite: int64   := TestArray.item<int64>([4]);

    var a    : TArray2D<Integer> :=   [ [1,2,3], [4,5,6] ] ;
    var given: TNDArray          :=  vNumpy.empty_like<Integer>(a);
    var sh   : Tnp_Shape         := given.shape;
    var dt   : TDtype            := given.dtype;

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

end;

function log(epochidx: Integer): Double;
begin
    frmMain.redtOutput.Lines.Add('test callback :' + IntToStr(epochidx));
end;

function pylog(self, args : PPyObject): PPyObject;cdecl;
var
  epochidx: Integer ;
  d       : Double;
begin
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
    NumPyTest;

    // keras test
    Test1;
    esempio_XOR;
    MergeExample ;
    ImplementCallback;
    MNIST_CNN
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

    var ite: Double   := y.item<double>([1,2]);

    redtOutput.Lines.Add(' learning');
    stopwatch :=  TStopwatch.StartNew;
    // Randomly initialize weights
    w1 := vNumpy.randn([D_in, HH]);
    w2 := vNumpy.randn([HH, D_out]);

    learning_rate := 1.0E-06;
    loss := Double.MaxValue;

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
        grad_h[TNDArray.opLess(h, 0)]   := vNumpy.asarray(0);
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
 callback : TCallback;
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

    callback := TEarlyStopping.Create ('val_loss', 3)  ;

    callback := TLearningRateScheduler.Create (pylog)  ;
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

    //Compile and train
    model.Compile(TStringOrInstance.Create( TAdam.Create ), 'binary_crossentropy',['accuracy']);

    var batch_size : Integer := 2;
    var history: THistory := model.Fit(x, y, @batch_size, 10,1);

    model.Summary;
   
    var logs := history.HistoryLogs;

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

    var model : TModel := TModel.Create ( [ input ] , [ output ]);

    //Load train data
    var x : TNDarray := TNumPy.npArray<Double>( [ [ 0, 0 ], [ 0, 1 ], [ 1, 0 ], [ 1, 1 ] ] );
    var y : TNDarray := TNumPy.npArray<Double>( [ 0, 1, 1, 0 ] );

    var input1  : TKInput    := TKInput.Create(tnp_shape.Create([32, 32,3]));
    var conv1   : TBaseLayer := TConv2D.Create(32, [4, 4], 'relu').&Set([input1]);
    var pool1   : TBaseLayer := TMaxPooling2D.Create([2, 2]).&Set([conv1]);
    var flatten1: TBaseLayer := TFlatten.Create.&Set([pool1]);

    var input2  : TKInput    := TKInput.Create(tnp_shape.Create([32, 32,3]));
    var conv2   : TBaseLayer := TConv2D.Create(16, [8, 8], 'relu').&Set([input2]);
    var pool2   : TBaseLayer := TMaxPooling2D.Create([2, 2]).&Set([conv2]);
    var flatten2: TBaseLayer := TFlatten.Create.&Set([pool2]);

    var merge : TConcatenate := TConcatenate.Create([flatten1, flatten2]);

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

    var callback : TCallback := TLearningRateScheduler.Create (pylog) ;

    var lossHistory : TCallback := TCallback.Custom('LossHistory', 'LossHistory.py');


    //Compile and train
    model.Compile(TStringOrInstance.Create( TAdam.Create ), 'binary_crossentropy', [ 'accuracy' ]);
    var batch_size: Integer  := 2;
    var history : THistory := model.Fit(x, y, @batch_size, 10, 1, [ callback{lossHistory} ]);

   // var customLosses: TArray<Double> := lossHistory.GetDoubleArray('losses');

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
    redtOutput.Lines.Add( IntToStr(x_train.shape[0]) + ' train samples');
    redtOutput.Lines.Add( IntToStr(x_test.shape[0]) + '  test  samples');

    // Convert class vectors to binary class matrices
    var Util : TUtil := TUtil.Create;
    y_train := Util.ToCategorical(y_train, @num_classes);
    y_test  := Util.ToCategorical(y_test, @num_classes);

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

    //Compile with loss, metrics and optimizer
    model.Compile(TStringOrInstance.Create(TAdadelta.Create), 'categorical_crossentropy', [ 'accuracy' ]);

    //Train the model
    model.Fit(x_train, y_train, @batch_size, epochs, 1,nil,0,[ x_test, y_test ]);

    //Score the model for performance
    var score : TArray<Double> := model.Evaluate(x_test, y_test, nil, 0);

    redtOutput.Lines.Add('Test loss: '   + FloatToStr(score[0]));
    redtOutput.Lines.Add('Test accuracy:'+ FloatToStr(score[1]));

    // Save the model to HDF5 format which can be loaded later or ported to other application
    model.Save('model.h5');
    // Save it to Tensorflow JS format and we will test it in browser.
    model.SaveTensorflowJSFormat('./');
end;

procedure TfrmMain.FormShow(Sender: TObject);
var
   jpg: TJpegImage;
   bmp: TBitmap;
begin
    // inizialige python engine with gui input/output
    InitGlobal(PyIOCom);

    vNumpy := TNumPy.Init(True);

    jpg := TJpegImage.Create;
    jpg.LoadFromFile('nn.jpg');
    img1.Picture.Assign(jpg);
end;

end.
