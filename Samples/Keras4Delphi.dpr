program Keras4Delphi;

uses
  Vcl.Forms,
  untMain in 'untMain.pas' {frmMain},
  np.Base in '..\src\NumPy\np.Base.pas',
  Python.Utils in '..\src\Python.Utils.pas',
  NDArray.Api in '..\src\NumPy\NDArray.Api.pas',
  np.Api in '..\src\NumPy\np.Api.pas',
  Models in '..\src\NumPy\Models.pas',
  Keras in '..\src\Keras.pas',
  Keras.Models in '..\src\Keras.Models.pas',
  utils in '..\src\utils.pas',
  Keras.Layers in '..\src\Keras.Layers.pas',
  Keras.PreProcessing in '..\src\Keras.PreProcessing.pas';

{$R *.res}

begin
  Application.Initialize;
  Application.MainFormOnTaskbar := True;
  Application.CreateForm(TfrmMain, frmMain);
  Application.Run;
end.
