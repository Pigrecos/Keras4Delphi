object frmMain: TfrmMain
  Left = 0
  Top = 0
  Caption = '[Demo test] Keras for Delphi'
  ClientHeight = 504
  ClientWidth = 835
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
  object splBottom: TSplitter
    Left = 0
    Top = 147
    Width = 835
    Height = 5
    Cursor = crVSplit
    Align = alBottom
    ExplicitLeft = -69
    ExplicitTop = 498
    ExplicitWidth = 904
  end
  object pnlTop: TPanel
    Left = 0
    Top = 0
    Width = 835
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
    Width = 835
    Height = 104
    Align = alClient
    Caption = 'pnl1'
    TabOrder = 1
  end
  object redtOutput: TRichEdit
    Left = 0
    Top = 152
    Width = 835
    Height = 352
    Align = alBottom
    Color = clInfoBk
    Font.Charset = ANSI_CHARSET
    Font.Color = clWindowText
    Font.Height = -11
    Font.Name = 'MS Sans Serif'
    Font.Style = []
    ParentFont = False
    ScrollBars = ssBoth
    TabOrder = 2
    WordWrap = False
    Zoom = 100
    OnChange = redtOutputChange
  end
  object PyIOCom: TPythonGUIInputOutput
    UnicodeIO = True
    RawOutput = False
    Output = redtOutput
    Left = 544
    Top = 3
  end
end
