unit Keras.PreProcessing;

interface
   uses  System.SysUtils, System.Generics.Collections,
         PythonEngine, Keras,Models,Python.Utils,
         Keras.Layers;

type
  TSequenceUtil = class(TBase)
     public
       caller : TPythonObject;
       constructor Create;
       function PadSequences(sequences : TNDarray ;
                             maxlen    : PInteger = nil;
                             dtype     : string = 'int32';
                             padding   : string = 'pre';
                             truncating: string = 'pre';
                             value     : Double= 0): TNDArray;

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

end.
