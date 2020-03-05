unit Keras.PreProcessing;

interface
   uses  System.SysUtils, System.Generics.Collections,
         PythonEngine, Keras,np.Models ,Python.Utils,
         Keras.Layers,
         np.Utils;

type
  // Sequence
  TSequenceUtil = class(TBase)
     public
       caller : TPythonObject;
       constructor Create;
       function PadSequences(sequences : TNDarray ;
                             maxlen    : PInteger = nil;
                             dtype     : string = 'int32';
                             padding   : string = 'pre';
                             truncating: string = 'pre';
                             value     : Double= 0.0): TNDArray;

       function SkipGrams(sequence        : TNDarray;
                          vocabulary_size : Integer;
                          window_size     : Integer= 4;
                          negative_samples: Double= 1.0;
                          shuffle         : Boolean= true;
                          categorical     : Boolean= false;
                          sampling_table  : TNDArray= nil;
                          seed            : PInteger= nil): TNDArray;

       function MakeSamplingTable(size: Integer; sampling_factor : Double= 1e-05) : TNDArray;
  end;

  //TextProcessing
  TTokenizer = class(TBase)
     public
       caller : TPythonObject;
       constructor Create; overload;
       constructor Create(num_words      : PInteger = nil;
                          filters        : string = '!\"#$%&()*+,-./:;<=>?@[\\]^_`{|}~'+#9#13#10;
                          lower          : Boolean = true;
                          split          : string= ' ';
                          char_level     : Boolean= false;
                          oov_token      : PInteger= nil;
                          document_count : Integer= 0);overload;

       procedure FitOnTexts(texts: TArray<string>);
       procedure FitOnSequences(sequences: TArray<TSequence>);
       function  TextsToSequences(texts: TArray<string>):TArray<TSequence>;
       function  SequencesToTexts(sequences: TArray<TSequence>):TArray<string>;
       function  TextsToMatrix(texts: TArray<string>; mode: string = 'binary'):TMatrix;
  end;

  TTextUtil = class(TBase)
     public
       caller : TPythonObject;
       constructor Create; overload;
       function HashingTrick(text          : string ;
                             n             : Integer;
                             hash_function : string = '';
                             filters       : string = '!\"#$%&()*+,-./:;<=>?@[\\]^_`{|}~'+#9#13#10;
                             lower         : Boolean= true;
                             split         : string= ' '): TArray<Integer>;

       function OneHot(text   : string ;
                       n      : Integer;
                       filters: string = '!\"#$%&()*+,-./:;<=>?@[\\]^_`{|}~'+#9#13#10;
                       lower  : Boolean= true;
                       split  : string= ' '): TArray2D<Integer>;

       function TextToWordSequence(text    : String ;
                                   filters : String = '!\"#$%&()*+,-./:;<=>?@[\\]^_`{|}~'+#9#13#10;
                                   lower   : Boolean= true;
                                   split   : String= ' '): TArray<string>;
  end;

  // SequenceProcessing
  TTimeseriesGenerator = class(TBase)
     public
       constructor Create(data          : TNDArray;
                          targets       : TNDArray;
                          length        : Integer;
                          sampling_rate : Integer= 1;
                          stride        : Integer = 1;
                          start_index   : Integer= 0;
                          end_index     : PInteger= nil;
                          shuffle       : Boolean= false;
                          reverse       : Boolean= false;
                          batch_size    : Integer= 128);

  end;

implementation
         uses System.Rtti;
{ TSequenceUtil }

constructor TSequenceUtil.Create;
begin
    inherited create;

    caller := GetKerasClassIstance('preprocessing.sequence');
end;

function TSequenceUtil.MakeSamplingTable(size: Integer; sampling_factor: Double): TNDArray;
begin
    Parameters.Clear;
    Parameters.Add( TPair<String,TValue>.Create('size',size) );
    Parameters.Add( TPair<String,TValue>.Create('sampling_factor',sampling_factor) );


    Result :=  TNDArray.Create( InvokeStaticMethod(caller,'make_sampling_table',Parameters) )
end;

function TSequenceUtil.PadSequences(sequences: TNDarray; maxlen: PInteger; dtype, padding, truncating: string; value: Double): TNDArray;
begin
    Parameters.Clear;
    Parameters.Add( TPair<String,TValue>.Create('sequences',sequences) );

    if maxlen <> nil then Parameters.Add( TPair<String,TValue>.Create('maxlen',maxlen^))
    else                  Parameters.Add( TPair<String,TValue>.Create('maxlen', TPythonObject.None ));

    Parameters.Add( TPair<String,TValue>.Create('dtype',dtype) );
    Parameters.Add( TPair<String,TValue>.Create('padding',padding) );
    Parameters.Add( TPair<String,TValue>.Create('truncating',truncating) );
    Parameters.Add( TPair<String,TValue>.Create('value',value) );

    Result :=  TNDArray.Create( InvokeStaticMethod(caller,'pad_sequences',Parameters) )
end;

function TSequenceUtil.SkipGrams(sequence: TNDarray; vocabulary_size, window_size: Integer; negative_samples: Double;
                                                shuffle, categorical: Boolean; sampling_table: TNDArray; seed: PInteger): TNDArray;
begin
    Parameters.Clear;
    Parameters.Add( TPair<String,TValue>.Create('sequences',sequence) );
    Parameters.Add( TPair<String,TValue>.Create('vocabulary_size',vocabulary_size) );
    Parameters.Add( TPair<String,TValue>.Create('window_size',window_size) );
    Parameters.Add( TPair<String,TValue>.Create('negative_samples',negative_samples) );
    Parameters.Add( TPair<String,TValue>.Create('shuffle',shuffle) );
    Parameters.Add( TPair<String,TValue>.Create('categorical',categorical) );

    if sampling_table <> nil then Parameters.Add( TPair<String,TValue>.Create('sampling_table',sampling_table))
    else                          Parameters.Add( TPair<String,TValue>.Create('sampling_table', TPythonObject.None ));

    if sampling_table <> nil then Parameters.Add( TPair<String,TValue>.Create('seed',seed^))
    else                          Parameters.Add( TPair<String,TValue>.Create('seed', TPythonObject.None ));

    Result :=  TNDArray.Create( InvokeStaticMethod(caller,'skipgrams',Parameters) )
end;

{ TTokenizer }

constructor TTokenizer.Create;
begin
    inherited create;

    caller := GetKerasClassIstance('preprocessing.text');
end;

constructor TTokenizer.Create(num_words: PInteger; filters: string; lower: Boolean; split: string; char_level: Boolean;
                                oov_token: PInteger; document_count: Integer);
begin
    Parameters.Clear;

    if num_words <> nil then Parameters.Add( TPair<String,TValue>.Create('num_words',num_words^))
    else                     Parameters.Add( TPair<String,TValue>.Create('num_words', TPythonObject.None ));

    Parameters.Add( TPair<String,TValue>.Create('filters',filters) );
    Parameters.Add( TPair<String,TValue>.Create('lower',lower) );
    Parameters.Add( TPair<String,TValue>.Create('split',split) );
    Parameters.Add( TPair<String,TValue>.Create('char_level',char_level) );

    if oov_token <> nil then Parameters.Add( TPair<String,TValue>.Create('oov_token',oov_token^))
    else                     Parameters.Add( TPair<String,TValue>.Create('oov_token', TPythonObject.None ));

    Parameters.Add( TPair<String,TValue>.Create('document_count',document_count) );

    PyInstance :=  InvokeStaticMethod(caller,'Tokenizer',Parameters)
end;

procedure TTokenizer.FitOnSequences(sequences: TArray<TSequence>);
begin
    Parameters.Clear;

    Parameters.Add( TPair<String,TValue>.Create('sequences',TValue.FromArray<TSequence>(sequences)) );

    InvokeStaticMethod(caller,'fit_on_sequences',Parameters)
end;

procedure TTokenizer.FitOnTexts(texts: TArray<string>);
begin
    Parameters.Clear;

    Parameters.Add( TPair<String,TValue>.Create('texts',TValue.FromArray<string>(texts)) );

    InvokeStaticMethod(caller,'fit_on_texts',Parameters)
end;

function TTokenizer.SequencesToTexts(sequences: TArray<TSequence>): TArray<string>;
begin
    Parameters.Clear;

    Parameters.Add( TPair<String,TValue>.Create('sequences',TValue.FromArray<TSequence>(sequences)) );

    Result := InvokeStaticMethod(caller,'sequences_to_texts',Parameters).AsArrayofString;
end;

function TTokenizer.TextsToMatrix(texts: TArray<string>; mode: string): TMatrix;
var
  atmp: TPythonObject;
begin
    Parameters.Clear;

    Parameters.Add( TPair<String,TValue>.Create('texts',TValue.FromArray<string>(texts)) );
    Parameters.Add( TPair<String,TValue>.Create('mode',mode) );

    atmp := InvokeStaticMethod(caller,'texts_to_matrix',Parameters) ;

    Result := TMatrix.Create( atmp.Handle  );

end;

function TTokenizer.TextsToSequences(texts: TArray<string>): TArray<TSequence>;
var
  atmp: TArray<TPythonObject>;
  i   : Integer;
begin
    Parameters.Clear;

    Parameters.Add( TPair<String,TValue>.Create('texts',TValue.FromArray<string>(texts)) );

    atmp := InvokeStaticMethod(caller,'texts_to_sequences',Parameters).AsArrayofPyObj ;

    for i := 0 to High(atmp) do
      Result := Result + [ TSequence.Create( atmp[i].Handle )  ];
end;

{ TTextUtil }

constructor TTextUtil.Create;
begin
    inherited create;

    caller := GetKerasClassIstance('preprocessing.text');
end;

function TTextUtil.HashingTrick(text: string; n: Integer; hash_function, filters: string; lower: Boolean;
                                      split: string): TArray<Integer>;
begin
    Parameters.Clear;

    Parameters.Add( TPair<String,TValue>.Create('text',text) );
    Parameters.Add( TPair<String,TValue>.Create('n',n) );
    Parameters.Add( TPair<String,TValue>.Create('hash_function',hash_function) );
    Parameters.Add( TPair<String,TValue>.Create('filters',filters) );
    Parameters.Add( TPair<String,TValue>.Create('lower',lower) );
    Parameters.Add( TPair<String,TValue>.Create('split',split) );

    Result := InvokeStaticMethod(caller,'hashing_trick',Parameters).AsArrayofInt;
end;

function TTextUtil.OneHot(text: string; n: Integer; filters: string; lower: Boolean; split: string): TArray2D<Integer>;
begin
    Parameters.Clear;

    Parameters.Add( TPair<String,TValue>.Create('text',text) );
    Parameters.Add( TPair<String,TValue>.Create('n',n) );
    Parameters.Add( TPair<String,TValue>.Create('filters',filters) );
    Parameters.Add( TPair<String,TValue>.Create('lower',lower) );
    Parameters.Add( TPair<String,TValue>.Create('split',split) );

    InvokeStaticMethod(caller,'one_hot',Parameters);
end;

function TTextUtil.TextToWordSequence(text: String; filters: String; lower: Boolean; split: String): TArray<string>;
begin
    Parameters.Clear;
    Result := [];
    Parameters.Add( TPair<String,TValue>.Create('text',AnsiString( text )) );
    Parameters.Add( TPair<String,TValue>.Create('filters',filters) );
    Parameters.Add( TPair<String,TValue>.Create('lower',lower) );
    Parameters.Add( TPair<String,TValue>.Create('split',split) );

    Result := InvokeStaticMethod(caller,'text_to_word_sequence',Parameters).AsArrayofString;
end;

{ TTimeseriesGenerator }

constructor TTimeseriesGenerator.Create(data, targets: TNDArray; length, sampling_rate, stride, start_index: Integer;
                                                   end_index: PInteger; shuffle, reverse: Boolean; batch_size: Integer);
begin
    inherited Create;

    Parameters.Add( TPair<String,TValue>.Create('data',data) );
    Parameters.Add( TPair<String,TValue>.Create('targets',targets) );
    Parameters.Add( TPair<String,TValue>.Create('length',length) );
    Parameters.Add( TPair<String,TValue>.Create('sampling_rate',sampling_rate) );
    Parameters.Add( TPair<String,TValue>.Create('stride',stride) );
    Parameters.Add( TPair<String,TValue>.Create('start_index',start_index) );

    if end_index <> nil then Parameters.Add( TPair<String,TValue>.Create('end_index',end_index^))
    else                     Parameters.Add( TPair<String,TValue>.Create('end_index', TPythonObject.None ));

    Parameters.Add( TPair<String,TValue>.Create('shuffle',shuffle) );
    Parameters.Add( TPair<String,TValue>.Create('reverse',reverse) );
    Parameters.Add( TPair<String,TValue>.Create('batch_size',batch_size) );

    PyInstance := GetKerasClassIstance('preprocessing.sequence.TimeseriesGenerator');
    Init;
end;

end.
