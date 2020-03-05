program Keras4Delphi;

uses
  Vcl.Forms,
  untMain in 'untMain.pas' {frmMain},
  np.Base in '..\src\NumPy\np.Base.pas',
  Python.Utils in '..\src\Python.Utils.pas',
  NDArray.Api in '..\src\NumPy\NDArray.Api.pas',
  np.Api in '..\src\NumPy\np.Api.pas',
  np.Models in '..\src\NumPy\np.Models.pas',
  Keras in '..\src\Keras.pas',
  Keras.Models in '..\src\Keras.Models.pas',
  np.Utils in '..\src\NumPy\np.Utils.pas',
  Keras.Layers in '..\src\Keras.Layers.pas',
  Keras.PreProcessing in '..\src\Keras.PreProcessing.pas';

{$R *.res}

begin
  Application.Initialize;
  Application.MainFormOnTaskbar := True;
  Application.CreateForm(TfrmMain, frmMain);
  Application.Run;
end.
