object frmMain: TfrmMain
  Left = 0
  Top = 0
  Caption = '[Demo test] Keras for Delphi'
  ClientHeight = 554
  ClientWidth = 804
  Color = clBtnFace
  Font.Charset = DEFAULT_CHARSET
  Font.Color = clWindowText
  Font.Height = -11
  Font.Name = 'Tahoma'
  Font.Style = []
  OldCreateOrder = False
  OnShow = FormShow
  PixelsPerInch = 96
  TextHeight = 13
  object spl1: TSplitter
    Left = 241
    Top = 43
    Width = 5
    Height = 258
    ExplicitLeft = 497
    ExplicitTop = 42
  end
  object spl2: TSplitter
    Left = 0
    Top = 301
    Width = 804
    Height = 5
    Cursor = crVSplit
    Align = alBottom
    ExplicitTop = 309
  end
  object pnlTop: TPanel
    Left = 0
    Top = 0
    Width = 804
    Height = 43
    Align = alTop
    TabOrder = 0
    object btn1: TButton
      Left = 16
      Top = 12
      Width = 75
      Height = 25
      Caption = 'keras'
      TabOrder = 0
      OnClick = btn1Click
    end
  end
  object pnl1: TPanel
    Left = 0
    Top = 43
    Width = 241
    Height = 258
    Align = alLeft
    Caption = 'pnl1'
    TabOrder = 1
    object img1: TImage
      Left = 1
      Top = 1
      Width = 239
      Height = 256
      Align = alClient
      Stretch = True
      ExplicitWidth = 232
    end
  end
  object redtOutput: TRichEdit
    Left = 0
    Top = 306
    Width = 804
    Height = 248
    Align = alBottom
    Color = clInfoBk
    Font.Charset = ANSI_CHARSET
    Font.Color = clWindowText
    Font.Height = -11
    Font.Name = 'MS Sans Serif'
    Font.Style = []
    ParentFont = False
    ScrollBars = ssVertical
    TabOrder = 2
    WordWrap = False
    Zoom = 100
    OnChange = redtOutputChange
  end
  object pnl2: TPanel
    Left = 246
    Top = 43
    Width = 558
    Height = 258
    Align = alClient
    Caption = 'pnl2'
    TabOrder = 3
    object cht1: TChart
      Left = 1
      Top = 1
      Width = 556
      Height = 256
      Legend.LegendStyle = lsSeries
      Legend.TextStyle = ltsPlain
      Title.Text.Strings = (
        'Training and Validation Accuracy')
      BottomAxis.Title.Caption = 'Epoch'
      LeftAxis.Title.Caption = 'Loss'
      Align = alClient
      TabOrder = 0
      DefaultCanvas = 'TGDIPlusCanvas'
      PrintMargins = (
        15
        19
        15
        19)
      ColorPaletteIndex = 18
      object srsTraining_Loss: TLineSeries
        Legend.Text = 'Training loss'
        LegendTitle = 'Training loss'
        Brush.BackColor = clDefault
        Pointer.InflateMargins = True
        Pointer.Style = psRectangle
        XValues.Name = 'X'
        XValues.Order = loAscending
        YValues.Name = 'Y'
        YValues.Order = loNone
      end
      object srsValidation_loss: TLineSeries
        Legend.Text = 'Validation accuracy'
        LegendTitle = 'Validation accuracy'
        Brush.BackColor = clDefault
        Pointer.InflateMargins = True
        Pointer.Style = psRectangle
        XValues.Name = 'X'
        XValues.Order = loAscending
        YValues.Name = 'Y'
        YValues.Order = loNone
      end
    end
  end
  object PyIOCom: TPythonGUIInputOutput
    UnicodeIO = True
    RawOutput = False
    Output = redtOutput
    Left = 544
    Top = 3
  end
end
