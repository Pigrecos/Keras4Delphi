{*******************************************************}
{                                                       }
{       Numpy Wrapper for Delphi                        }
{                                                       }
{       Copyright (C) 2020 Pigrecos(Max)                }
{                                                       }
{*******************************************************}
unit np.Base;

// np.dtype.gen.cs
// np.module.gen.cs
// TDtypeExtensions  -----> for circular reference error

interface
   uses Winapi.Windows, System.SysUtils,System.Rtti,
        PythonEngine,
        Python.Utils,

        utils,
        Models;

type
  Constants = (inf, neg_inf) ;

  PTPythonObject = ^TPythonObject;
  PTNDarray      = TArray<TNDarray> ;
  PPTNDarray     = ^PTNDarray;

  TNumPy = class;



  TDtypeExtensions = class
   public
     class function GetDtype<T>(pObj:T):TDtype;
  end;

  TNumPy = class
  private
    class function Getbool_: TDtype;
    class function Getbool8: TDtype;
    class function Getbyte: TDtype;
    class function Getbytes_: TDtype;
    class function Getclongfloat: TDtype;
    class function Getcomplex_: TDtype;
    class function Getcomplex128: TDtype;
    class function Getcomplex192: TDtype;
    class function Getcomplex256: TDtype;
    class function Getcomplex64: TDtype;
    class function Getcsingle: TDtype;
    class function Getdouble: TDtype;
    class function Getfloat_: TDtype;
    class function Getfloat128: TDtype;
    class function Getfloat16: TDtype;
    class function Getfloat32: TDtype;
    class function Getfloat64: TDtype;
    class function Getfloat96: TDtype;
    class function Gethalf: TDtype;
    class function Getint_: TDtype;
    class function Getint16: TDtype;
    class function Getint32: TDtype;
    class function Getint64: TDtype;
    class function Getint8: TDtype;
    class function Getintc: TDtype;
    class function Getintp: TDtype;
    class function Getlongfloat: TDtype;
    class function Getlonglong: TDtype;
    class function Getobject_: TDtype;
    class function Getshort: TDtype;
    class function Getsingle: TDtype;
    class function Getubyte: TDtype;
    class function Getuint: TDtype;
    class function Getuint16: TDtype;
    class function Getuint32: TDtype;
    class function Getuint64: TDtype;
    class function Getuint8: TDtype;
    class function Getuintc: TDtype;
    class function Getuintp: TDtype;
    class function Getulonglong: TDtype;
    class function Getunicode_: TDtype;
    class function Getushort: TDtype;
    class function Getvoid: TDtype;
    procedure Setbool_(const Value: TDtype);
    procedure Setbool8(const Value: TDtype);
    procedure Setbyte(const Value: TDtype);
    procedure Setbytes_(const Value: TDtype);
    procedure Setclongfloat(const Value: TDtype);
    procedure Setcomplex_(const Value: TDtype);
    procedure Setcomplex128(const Value: TDtype);
    procedure Setcomplex192(const Value: TDtype);
    procedure Setcomplex256(const Value: TDtype);
    procedure Setcomplex64(const Value: TDtype);
    procedure Setcsingle(const Value: TDtype);
    procedure Setdouble(const Value: TDtype);
    procedure Setfloat_(const Value: TDtype);
    procedure Setfloat128(const Value: TDtype);
    procedure Setfloat16(const Value: TDtype);
    procedure Setfloat32(const Value: TDtype);
    procedure Setfloat64(const Value: TDtype);
    procedure Setfloat96(const Value: TDtype);
    procedure Sethalf(const Value: TDtype);
    procedure Setint_(const Value: TDtype);
    procedure Setint16(const Value: TDtype);
    procedure Setint32(const Value: TDtype);
    procedure Setint64(const Value: TDtype);
    procedure Setint8(const Value: TDtype);
    procedure Setintc(const Value: TDtype);
    procedure Setintp(const Value: TDtype);
    procedure Setlongfloat(const Value: TDtype);
    procedure Setlonglong(const Value: TDtype);
    procedure Setobject_(const Value: TDtype);
    procedure Setshort(const Value: TDtype);
    procedure Setsingle(const Value: TDtype);
    procedure Setubyte(const Value: TDtype);
    procedure Setuint(const Value: TDtype);
    procedure Setuint16(const Value: TDtype);
    procedure Setuint32(const Value: TDtype);
    procedure Setuint64(const Value: TDtype);
    procedure Setuint8(const Value: TDtype);
    procedure Setuintc(const Value: TDtype);
    procedure Setuintp(const Value: TDtype);
    procedure Setulonglong(const Value: TDtype);
    procedure Setunicode_(const Value: TDtype);
    procedure Setushort(const Value: TDtype);
    procedure Setvoid(const Value: TDtype);
    // np.module.gen.cs
    public
      class var FhModuleNumPy : TPythonObject;

      constructor Init(inizialize: Boolean);

      class function  ToTuple: TPyTuple;overload; static;
      class function  ToTuple(input: TArray<TValue>):  TPyTuple; overload; static;
      class function  ToPython(value:TValue):          TPythonObject;overload; static;
      class function  ToCsharp<T>(pyobj:TPythonObject): T;  static;
      class function  ConvertArrayToNDarray<T>(a: TArray<T>): TNDarray; overload;
      class function  ConvertArrayToNDarray<T>(a: TArray2D<T>): TNDarray; overload;

      function inf         : Double;
      function Infinity    : Double;
      function PINF        : Double;
      function infty       : Double;
      function NINF        : Double;
      function nan         : Double;
      function NZERO       : Double;
      function PZERO       : Double;
      function e           : Double;
      function euler_gamma : Double;
      function newaxis     : Double;
      function pi          : Double;

      property bool_ :     TDtype  read Getbool_      write Setbool_;
      property bool8_:     TDtype  read Getbool8      write Setbool8;
      property byte_ :     TDtype  read Getbyte       write Setbyte;
      property short_:     TDtype  read Getshort      write Setshort;
      property intc_:      TDtype  read Getintc       write Setintc;
      property int_ :      TDtype  read Getint_       write Setint_;
      property longlong_:  TDtype  read Getlonglong   write Setlonglong;
      property intp_:      TDtype  read Getintp       write Setintp;
      property int8_:      TDtype  read Getint8       write Setint8;
      property int16_:     TDtype  read Getint16      write Setint16;
      property int32_:     TDtype  read Getint32      write Setint32;
      property int64_:     TDtype  read Getint64      write Setint64;
      property ubyte_:     TDtype  read Getubyte      write Setubyte;
      property ushort_:    TDtype  read Getushort     write Setushort;
      property uintc_:     TDtype  read Getuintc      write Setuintc;
      property uint_:      TDtype  read Getuint       write Setuint;
      property ulonglong_: TDtype  read Getulonglong  write Setulonglong;
      property uintp_:     TDtype  read Getuintp      write Setuintp;
      property uint8_:     TDtype  read Getuint8      write Setuint8;
      property uint16_:    TDtype  read Getuint16     write Setuint16;
      property uint32_:    TDtype  read Getuint32     write Setuint32;
      property uint64_:    TDtype  read Getuint64     write Setuint64;
      property half_:      TDtype  read Gethalf       write Sethalf;
      property single_:    TDtype  read Getsingle     write Setsingle;
      property double_:    TDtype  read Getdouble     write Setdouble;
      property float_ :    TDtype  read Getfloat_     write Setfloat_;
      property longfloat_: TDtype  read Getlongfloat  write Setlongfloat;
      property float16_:   TDtype  read Getfloat16    write Setfloat16;
      property float32_:   TDtype  read Getfloat32    write Setfloat32;
      property float64_:   TDtype  read Getfloat64    write Setfloat64;
      property float96_:   TDtype  read Getfloat96    write Setfloat96;
      property float128_:  TDtype  read Getfloat128   write Setfloat128;
      property csingle_:   TDtype  read Getcsingle    write Setcsingle;
      property complex_ :  TDtype  read Getcomplex_   write Setcomplex_;
      property clongfloat_:TDtype  read Getclongfloat write Setclongfloat;
      property complex64_: TDtype  read Getcomplex64  write Setcomplex64;
      property complex128_:TDtype  read Getcomplex128 write Setcomplex128;
      property complex192_:TDtype  read Getcomplex192 write Setcomplex192;
      property complex256_:TDtype  read Getcomplex256 write Setcomplex256;
      property object_ :   TDtype  read Getobject_    write Setobject_;
      property bytes_ :    TDtype  read Getbytes_     write Setbytes_;
      property unicode_ :  TDtype  read Getunicode_   write Setunicode_;
      property void_ :     TDtype  read Getvoid       write Setvoid;

      //property hModuleNumPy : TPythonObject  read FhModuleNumPy;


  end;

implementation
    uses
       System.Generics.Collections,
       System.TypInfo,
       np.Api;

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

{ TDtypeExtensions }

class function TDtypeExtensions.GetDtype<T>(pObj: T): TDtype;
var
  Info     : PTypeInfo;
  res      : TPythonObject;
  rtti     : TRttiType;
begin

    { Get type info for the "yet unknown type" }
    Info := TypeInfo(T);

    if       System.TypeInfo(T) = System.TypeInfo(boolean)  then  Result := TNumPy.Getbool8

    else if  System.TypeInfo(T) = System.TypeInfo(int8)     then  Result := TNumPy.Getint8
    else if  System.TypeInfo(T) = System.TypeInfo(UInt8)    then  Result := TNumPy.Getuint8
    else if  System.TypeInfo(T) = System.TypeInfo(Byte)     then  Result := TNumPy.Getuint8

    else if  System.TypeInfo(T) = System.TypeInfo(int16)    then  Result := TNumPy.Getint16
    else if  System.TypeInfo(T) = System.TypeInfo(Word)     then  Result := TNumPy.Getuint16
    else if  System.TypeInfo(T) = System.TypeInfo(UInt16)   then  Result := TNumPy.Getuint16

    else if  System.TypeInfo(T) = System.TypeInfo(Int32)    then  Result := TNumPy.Getint32
    else if  System.TypeInfo(T) = System.TypeInfo(Cardinal) then  Result := TNumPy.Getuint32
    else if  System.TypeInfo(T) = System.TypeInfo(Uint32)   then  Result := TNumPy.Getuint32

    else if  System.TypeInfo(T) = System.TypeInfo(Int64)    then  Result := TNumPy.Getint64
    else if  System.TypeInfo(T) = System.TypeInfo(uint64)   then  Result := TNumPy.Getuint64

    else if  System.TypeInfo(T) = System.TypeInfo(Single)   then  Result := TNumPy.Getfloat32
    else if  System.TypeInfo(T) = System.TypeInfo(double)   then  Result := TNumPy.Getfloat64

    else if  System.TypeInfo(T) = System.TypeInfo(string)   then  Result := TNumPy.Getunicode_
    else if  System.TypeInfo(T) = System.TypeInfo(char)     then  Result := TNumPy.Getunicode_

    else if (Info.Kind = tkArray) then
    begin
        rtti := TRttiContext.Create.GetType(TypeInfo(T));

        if       TRttiArrayType(rtti).Handle = System.TypeInfo(TArray<boolean>)  then  Result := TNumPy.Getbool8
                                                                                                 
        else if  TRttiArrayType(rtti).Handle = System.TypeInfo(TArray<int8>)     then  Result := TNumPy.Getint8
        else if  TRttiArrayType(rtti).Handle = System.TypeInfo(TArray<UInt8>)    then  Result := TNumPy.Getuint8
        else if  TRttiArrayType(rtti).Handle = System.TypeInfo(TArray<Byte>)     then  Result := TNumPy.Getuint8
                                                                                                 
        else if  TRttiArrayType(rtti).Handle = System.TypeInfo(TArray<int16>)    then  Result := TNumPy.Getint16
        else if  TRttiArrayType(rtti).Handle = System.TypeInfo(TArray<Word>)     then  Result := TNumPy.Getuint16
        else if  TRttiArrayType(rtti).Handle = System.TypeInfo(TArray<UInt16>)   then  Result := TNumPy.Getuint16
                                                                                                 
        else if  TRttiArrayType(rtti).Handle = System.TypeInfo(TArray<Int32>)    then  Result := TNumPy.Getint32
        else if  TRttiArrayType(rtti).Handle = System.TypeInfo(TArray<Cardinal>) then  Result := TNumPy.Getuint32
        else if  TRttiArrayType(rtti).Handle = System.TypeInfo(TArray<Uint32>)   then  Result := TNumPy.Getuint32
                                                                                                 
        else if  TRttiArrayType(rtti).Handle = System.TypeInfo(TArray<Int64>)    then  Result := TNumPy.Getint64
        else if  TRttiArrayType(rtti).Handle = System.TypeInfo(TArray<uint64>)   then  Result := TNumPy.Getuint64
                                                                                                 
        else if  TRttiArrayType(rtti).Handle = System.TypeInfo(TArray<Single>)   then  Result := TNumPy.Getfloat32
        else if  TRttiArrayType(rtti).Handle = System.TypeInfo(TArray<double>)   then  Result := TNumPy.Getfloat64
                                                                                                 
        else if  TRttiArrayType(rtti).Handle = System.TypeInfo(TArray<string>)   then  Result := TNumPy.Getunicode_
        else if  TRttiArrayType(rtti).Handle = System.TypeInfo(TArray<char>)     then  Result := TNumPy.Getunicode_
    end
    else if (Info.Kind = tkDynArray) then
    begin
        rtti := TRttiContext.Create.GetType(TypeInfo(T));

        if       TRttiDynamicArrayType(rtti).Handle = System.TypeInfo(TArray<boolean>)  then  Result := TNumPy.Getbool8
                                                                                                        
        else if  TRttiDynamicArrayType(rtti).Handle = System.TypeInfo(TArray<int8>)     then  Result := TNumPy.Getint8
        else if  TRttiDynamicArrayType(rtti).Handle = System.TypeInfo(TArray<UInt8>)    then  Result := TNumPy.Getuint8
        else if  TRttiDynamicArrayType(rtti).Handle = System.TypeInfo(TArray<Byte>)     then  Result := TNumPy.Getuint8
                                                                                                        
        else if  TRttiDynamicArrayType(rtti).Handle = System.TypeInfo(TArray<int16>)    then  Result := TNumPy.Getint16
        else if  TRttiDynamicArrayType(rtti).Handle = System.TypeInfo(TArray<Word>)     then  Result := TNumPy.Getuint16
        else if  TRttiDynamicArrayType(rtti).Handle = System.TypeInfo(TArray<UInt16>)   then  Result := TNumPy.Getuint16
                                                                                                        
        else if  TRttiDynamicArrayType(rtti).Handle = System.TypeInfo(TArray<Int32>)    then  Result := TNumPy.Getint32
        else if  TRttiDynamicArrayType(rtti).Handle = System.TypeInfo(TArray<Cardinal>) then  Result := TNumPy.Getuint32
        else if  TRttiDynamicArrayType(rtti).Handle = System.TypeInfo(TArray<Uint32>)   then  Result := TNumPy.Getuint32
                                                                                                        
        else if  TRttiDynamicArrayType(rtti).Handle = System.TypeInfo(TArray<Int64>)    then  Result := TNumPy.Getint64
        else if  TRttiDynamicArrayType(rtti).Handle = System.TypeInfo(TArray<uint64>)   then  Result := TNumPy.Getuint64
                                                                                                        
        else if  TRttiDynamicArrayType(rtti).Handle = System.TypeInfo(TArray<Single>)   then  Result := TNumPy.Getfloat32
        else if  TRttiDynamicArrayType(rtti).Handle = System.TypeInfo(TArray<double>)   then  Result := TNumPy.Getfloat64
                                                                                                        
        else if  TRttiDynamicArrayType(rtti).Handle = System.TypeInfo(TArray<string>)   then  Result := TNumPy.Getunicode_
        else if  TRttiDynamicArrayType(rtti).Handle = System.TypeInfo(TArray<char>)     then  Result := TNumPy.Getunicode_
    end

    else
      raise EPythonError.Create('Can not convert type of given object to dtype: " ');

end;

{ TNumPy }

constructor TNumPy.Init(inizialize: Boolean);
begin
    //Self := default(TNumPy) ;
    if  g_MyPyEngine = nil then
      InitGlobal;

    MaskFPUExceptions(True);
    if FhModuleNumPy = nil then
      FhModuleNumPy :=  TPythonObject.Create( ImportModule('numpy') );

end;

class function TNumPy.ToCsharp<T>(pyobj:TPythonObject): T;
var
  Info     : PTypeInfo;
  Data     : PTypeData;

  p         : PPyObject;
  rv        : TArray<TNDarray>;
  Name      : string;
  seq_length, i: Integer;
begin
    { Get type info for the "yet unknown type" }
    Info := System.TypeInfo(T);
    Data := GetTypeData(Info);

    if info^.Kind = tkClass then
    begin
        Name := string(Info^.Name);
        // types from 'ToCsharpConversions'
        if      LowerCase(Name) = 'tdtype'     then  PTPythonObject(@Result)^ := TDtype.Create(pyobj.Handle)
        else if LowerCase(Name) = 'tndarray'   then  PTPythonObject(@Result)^ := TNDArray.Create(pyobj)
        else if LowerCase(Name) = 'tmatrix'    then  PTPythonObject(@Result)^ := TMatrix.Create(pyobj.Handle)
        else if LowerCase(Name).Contains('tndarray<') then
        begin
            if       LowerCase(Name).Contains('byte')       then PTPythonObject(@Result)^ := TNDArray<Byte>.Create(pyobj)
            else if  LowerCase(Name).Contains('word')       then PTPythonObject(@Result)^ := TNDArray<Word>.Create(pyobj)
            else if  LowerCase(Name).Contains('boolean')    then PTPythonObject(@Result)^ := TNDArray<Boolean>.Create(pyobj)
            else if  (LowerCase(Name).Contains('int32')) or
                     (LowerCase(Name).Contains('integer'))  then PTPythonObject(@Result)^ := TNDArray<Int32>.Create(pyobj)
            else if  LowerCase(Name).Contains('int64')      then PTPythonObject(@Result)^ := TNDArray<Int64>.Create(pyobj)
            else if  (LowerCase(Name).Contains('single')) or
                     (LowerCase(Name).Contains('float32'))  then PTPythonObject(@Result)^ := TNDArray<Float32>.Create(pyobj)
            else if  LowerCase(Name).Contains('double')     then PTPythonObject(@Result)^ := TNDArray<Double>.Create(pyobj)
            else
               raise Exception.Create('Type ' + Name + 'missing. Add it to "ToCsharpConversions"');
        end;
    end
    else if info^.Kind = tkRecord then
    begin
        Name := string(Info^.Name);
        // types from 'ToCsharpConversions'
        if LowerCase(Name) = 'tndarray'   then  PTPythonObject(@Result)^ := TNDArray.Create(pyobj)
        else if LowerCase(Name).Contains('tndarray<') then
        begin
            if       LowerCase(Name).Contains('byte')       then PTPythonObject(@Result)^ := TNDArray<Byte>.Create(pyobj)
            else if  LowerCase(Name).Contains('word')       then PTPythonObject(@Result)^ := TNDArray<Word>.Create(pyobj)
            else if  LowerCase(Name).Contains('boolean')    then PTPythonObject(@Result)^ := TNDArray<Boolean>.Create(pyobj)
            else if  (LowerCase(Name).Contains('int32')) or
                     (LowerCase(Name).Contains('integer'))  then PTPythonObject(@Result)^ := TNDArray<Int32>.Create(pyobj)
            else if  LowerCase(Name).Contains('int64')      then PTPythonObject(@Result)^ := TNDArray<Int64>.Create(pyobj)
            else if  (LowerCase(Name).Contains('single')) or
                     (LowerCase(Name).Contains('float32'))  then PTPythonObject(@Result)^ := TNDArray<Float32>.Create(pyobj)
            else if  LowerCase(Name).Contains('double')     then PTPythonObject(@Result)^ := TNDArray<Double>.Create(pyobj)
            else
               raise Exception.Create('Type ' + Name + 'missing. Add it to "ToCsharpConversions"');
        end;
    end
    else if (Info^.Kind = tkArray) or (Info^.Kind = tkDynArray) then
    begin
        if g_MyPyEngine.PySequence_Check( pyobj.Handle ) = 1 then
        begin
             seq_length := g_MyPyEngine.PySequence_Length( pyobj.Handle );
             // if we have at least one object in the sequence,
             if seq_length > 0 then
                // we try to get the first one, simply to test if the sequence API is really implemented.
                g_MyPyEngine.Py_XDecRef( g_MyPyEngine.PySequence_GetItem( pyobj.Handle, 0 ) );
             if g_MyPyEngine.PyErr_Occurred = nil then
             begin
                 for i := 0 to g_MyPyEngine.PySequence_Length( pyobj.Handle ) -1 do
                 begin
                     p := g_MyPyEngine.PySequence_GetItem( pyobj.Handle, i );
                     rv := rv + [ ToCsharp<TNDarray>( TPythonObject.Create(p) ) ];
                 end;

                 PPTNDarray(@Result)^  := rv;
             end else
             begin
                 g_MyPyEngine.PyErr_Clear;
                 PTPythonObject(@Result)^ := nil;
             end;
        end;
    end else
    begin
       if      (Info^.Kind = tkInteger) then
       begin
           case data^.OrdType of
             otUByte,otSByte: PByte(@Result)^         :=   g_MyPyEngine.PyObjectAsVariant(pyobj.Handle);
             otUWord,otSWord: PWord(@Result)^         :=   g_MyPyEngine.PyObjectAsVariant(pyobj.Handle);
             otULong,otSLong: PInteger(@Result)^      :=   g_MyPyEngine.PyObjectAsVariant(pyobj.Handle);
           end;
       end
       else if (Info^.Kind = tkInt64)           then  PInt64(@Result)^           :=   g_MyPyEngine.PyObjectAsVariant(pyobj.Handle)
       else if (Info^.Kind = tkFloat)           then  PFloat(@Result)^           :=   g_MyPyEngine.PyObjectAsVariant(pyobj.Handle)
       else if (Info^.Kind = tkUnicodeString)   then  PUnicodeString(@Result)^   :=   g_MyPyEngine.PyObjectAsVariant(pyobj.Handle)
       else if (Info^.Kind = tkWideString)      then  PWideString(@Result)^      :=   g_MyPyEngine.PyObjectAsVariant(pyobj.Handle)
       else if (Info^.Kind = tkAnsiString   )   then  PAnsiString(@Result)^      :=   g_MyPyEngine.PyObjectAsVariant(pyobj.Handle)
       else if (Info = System.TypeInfo(boolean))then  PBoolean(@Result)^         :=   g_MyPyEngine.PyObjectAsVariant(pyobj.Handle)
    end;
end;

class function TNumPy.ToPython(value: TValue): TPythonObject;
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
        if (data.ParentInfo^.Name = 'TPythonObject') or (data.ParentInfo^.Name = 'TNDArray') then
           Result := value.AsType<TPythonObject>
        // TList<System.string>
        else if Info.Name = 'TList<System.string>' then
        begin
            var aTmpStr    : TArray<string>;
            var i          : Integer;

            aTmpStr := TList<string>(value.AsObject).ToArray;
            for i := 0 to High(aTmpStr) do
              aTmp := aTmp + [ aTmpStr[i] ];

            Result := TPythonObject.Create( g_MyPyEngine.ArrayToPyList( TValueArrayToArrayOfConst( aTmp ) ) );
        end;
    end
    // array type
    else if (Info.Kind = tkDynArray)  or (Info.Kind = tkArray) then
    begin
        Result := ToTuple( value.AsArray );
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
    else if Info = System.TypeInfo(string)         then
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

class function TNumPy.ConvertArrayToNDarray<T>(a: TArray<T>): TNDarray;
begin
    Result := npArray<T>(a);
end;

class function TNumPy.ConvertArrayToNDarray<T>(a: TArray2D<T>): TNDarray;
begin
    Result := npArray<T>(a).reshape( [ Length(a), Length(a[0]) ])
end;

class function TNumPy.ToTuple(): TPyTuple;
var
 aArray : TArray<TPythonObject>;

begin
    SetLength(aArray,0) ;

    Result := TPyTuple.Create(aArray);
end;

class function TNumPy.ToTuple(input: TArray<TValue>): TPyTuple;
var
 aArray : TArray<TPythonObject>;
 i : Integer;
begin
    SetLength(aArray,Length(input)) ;
    for i := 0 to  Length(input) - 1 do
     aArray[i] := ToPython(input[i]);

    Result := TPyTuple.Create(aArray);
end;

class function TNumPy.Getbool8: TDtype;
begin
    Result := ToCsharp<TDtype>( FhModuleNumPy.GetAttr('bool8'))
end;

class function TNumPy.Getbool_: TDtype;
begin
    Result := ToCsharp<TDtype>( FhModuleNumPy.GetAttr('bool_'))
end;

class function TNumPy.Getbyte: TDtype;
begin
    Result := ToCsharp<TDtype>( FhModuleNumPy.GetAttr('byte'))
end;

class function TNumPy.Getbytes_: TDtype;
begin
    Result := ToCsharp<TDtype>( FhModuleNumPy.GetAttr('bytes_'))
end;

class function TNumPy.Getclongfloat: TDtype;
begin
    Result := ToCsharp<TDtype>( FhModuleNumPy.GetAttr('clongfloat'))
end;

class function TNumPy.Getcomplex128: TDtype;
begin
    Result := ToCsharp<TDtype>( FhModuleNumPy.GetAttr('complex128'))
end;

class function TNumPy.Getcomplex192: TDtype;
begin
    Result := ToCsharp<TDtype>( FhModuleNumPy.GetAttr('complex192'))
end;

class function TNumPy.Getcomplex256: TDtype;
begin
    Result := ToCsharp<TDtype>( FhModuleNumPy.GetAttr('complex256'))
end;

class function TNumPy.Getcomplex64: TDtype;
begin
    Result := ToCsharp<TDtype>( FhModuleNumPy.GetAttr('complex64'))
end;

class function TNumPy.Getcomplex_: TDtype;
begin
    Result := ToCsharp<TDtype>( FhModuleNumPy.GetAttr('complex_'))
end;

class function TNumPy.Getcsingle: TDtype;
begin
    Result := ToCsharp<TDtype>( FhModuleNumPy.GetAttr('csingle'))
end;

class function TNumPy.Getdouble: TDtype;
begin
   Result := ToCsharp<TDtype>( FhModuleNumPy.GetAttr('double'))
end;

class function TNumPy.Getfloat128: TDtype;
begin
    Result := ToCsharp<TDtype>( FhModuleNumPy.GetAttr('loat128'))
end;

class function TNumPy.Getfloat16: TDtype;
begin
   Result := ToCsharp<TDtype>( FhModuleNumPy.GetAttr('float16'))
end;

class function TNumPy.Getfloat32: TDtype;
begin
    Result := ToCsharp<TDtype>( FhModuleNumPy.GetAttr('float32'))
end;

class function TNumPy.Getfloat64: TDtype;
begin
   Result := ToCsharp<TDtype>( FhModuleNumPy.GetAttr('float64'))
end;

class function TNumPy.Getfloat96: TDtype;
begin
    Result := ToCsharp<TDtype>( FhModuleNumPy.GetAttr('float96'))
end;

class function TNumPy.Getfloat_: TDtype;
begin
   Result := ToCsharp<TDtype>( FhModuleNumPy.GetAttr('float_'))
end;

class function TNumPy.Gethalf: TDtype;
begin
   Result := ToCsharp<TDtype>( FhModuleNumPy.GetAttr('half'))
end;

class function TNumPy.Getint16: TDtype;
begin
   Result := ToCsharp<TDtype>( FhModuleNumPy.GetAttr('int16'))
end;

class function TNumPy.Getint32: TDtype;
begin
   Result := ToCsharp<TDtype>( FhModuleNumPy.GetAttr('int32'))
end;

class function TNumPy.Getint64: TDtype;
begin
   Result := ToCsharp<TDtype>( FhModuleNumPy.GetAttr('int64'))
end;

class function TNumPy.Getint8: TDtype;
begin
   Result := ToCsharp<TDtype>( FhModuleNumPy.GetAttr('int8'))
end;

class function TNumPy.Getintc: TDtype;
begin
   Result := ToCsharp<TDtype>( FhModuleNumPy.GetAttr('intc'))
end;

class function TNumPy.Getintp: TDtype;
begin
    Result := ToCsharp<TDtype>( FhModuleNumPy.GetAttr('intp'))
end;

class function TNumPy.Getint_: TDtype;
begin
   Result := ToCsharp<TDtype>( FhModuleNumPy.GetAttr('int_'))
end;

class function TNumPy.Getlongfloat: TDtype;
begin
   Result := ToCsharp<TDtype>( FhModuleNumPy.GetAttr('longfloat'))
end;

class function TNumPy.Getlonglong: TDtype;
begin
   Result := ToCsharp<TDtype>( FhModuleNumPy.GetAttr('longlong'))
end;

class function TNumPy.Getobject_: TDtype;
begin
    Result := ToCsharp<TDtype>( FhModuleNumPy.GetAttr('object_'))
end;

class function TNumPy.Getshort: TDtype;
begin
   Result := ToCsharp<TDtype>( FhModuleNumPy.GetAttr('short'))
end;

class function TNumPy.Getsingle: TDtype;
begin
   Result := ToCsharp<TDtype>( FhModuleNumPy.GetAttr('single'))
end;

class function TNumPy.Getubyte: TDtype;
begin
   Result := ToCsharp<TDtype>( FhModuleNumPy.GetAttr('ubyte'))
end;

class function TNumPy.Getuint: TDtype;
begin
   Result := ToCsharp<TDtype>( FhModuleNumPy.GetAttr('uint'))
end;

class function TNumPy.Getuint16: TDtype;
begin
   Result := ToCsharp<TDtype>( FhModuleNumPy.GetAttr('uint16'))
end;

class function TNumPy.Getuint32: TDtype;
begin
    Result := ToCsharp<TDtype>( FhModuleNumPy.GetAttr('uint32'))
end;

class function TNumPy.Getuint64: TDtype;
begin
   Result := ToCsharp<TDtype>( FhModuleNumPy.GetAttr('uint64'))
end;

class function TNumPy.Getuint8: TDtype;
begin
   Result := ToCsharp<TDtype>( FhModuleNumPy.GetAttr('uint8'))
end;

class function TNumPy.Getuintc: TDtype;
begin
    Result := ToCsharp<TDtype>( FhModuleNumPy.GetAttr('uintc'))
end;

class function TNumPy.Getuintp: TDtype;
begin
   Result := ToCsharp<TDtype>( FhModuleNumPy.GetAttr('uintp'))
end;

class function TNumPy.Getulonglong: TDtype;
begin
   Result := ToCsharp<TDtype>( FhModuleNumPy.GetAttr('ulonglong'))
end;

class function TNumPy.Getunicode_: TDtype;
begin
    Result := ToCsharp<TDtype>( FhModuleNumPy.GetAttr('unicode_'))
end;

class function TNumPy.Getushort: TDtype;
begin
   Result := ToCsharp<TDtype>( FhModuleNumPy.GetAttr('ushort'))
end;

class function TNumPy.Getvoid: TDtype;
begin
   Result := ToCsharp<TDtype>( FhModuleNumPy.GetAttr('void'))
end;

procedure TNumPy.Setbool8(const Value: TDtype);
begin
    FhModuleNumPy.SetAttr('bool8', ToPython(value));
end;

procedure TNumPy.Setbool_(const Value: TDtype);
begin
    FhModuleNumPy.SetAttr('bool_', ToPython(value));
end;

procedure TNumPy.Setbyte(const Value: TDtype);
begin
    FhModuleNumPy.SetAttr('byte', ToPython(value));
end;

procedure TNumPy.Setbytes_(const Value: TDtype);
begin
    FhModuleNumPy.SetAttr('bytes_', ToPython(value));
end;

procedure TNumPy.Setclongfloat(const Value: TDtype);
begin
    FhModuleNumPy.SetAttr('clongfloat', ToPython(value));
end;

procedure TNumPy.Setcomplex128(const Value: TDtype);
begin
    FhModuleNumPy.SetAttr('complex128', ToPython(value));
end;

procedure TNumPy.Setcomplex192(const Value: TDtype);
begin
    FhModuleNumPy.SetAttr('complex192', ToPython(value));
end;

procedure TNumPy.Setcomplex256(const Value: TDtype);
begin
    FhModuleNumPy.SetAttr('complex256', ToPython(value));
end;

procedure TNumPy.Setcomplex64(const Value: TDtype);
begin
    FhModuleNumPy.SetAttr('complex64', ToPython(value));
end;

procedure TNumPy.Setcomplex_(const Value: TDtype);
begin
    FhModuleNumPy.SetAttr('complex_', ToPython(value));
end;

procedure TNumPy.Setcsingle(const Value: TDtype);
begin
    FhModuleNumPy.SetAttr('csingle', ToPython(value));
end;

procedure TNumPy.Setdouble(const Value: TDtype);
begin
    FhModuleNumPy.SetAttr('double', ToPython(value));
end;

procedure TNumPy.Setfloat128(const Value: TDtype);
begin
    FhModuleNumPy.SetAttr('float128', ToPython(value));
end;

procedure TNumPy.Setfloat16(const Value: TDtype);
begin
   FhModuleNumPy.SetAttr('float16', ToPython(value));
end;

procedure TNumPy.Setfloat32(const Value: TDtype);
begin
    FhModuleNumPy.SetAttr('float32', ToPython(value));
end;

procedure TNumPy.Setfloat64(const Value: TDtype);
begin
    FhModuleNumPy.SetAttr('float64', ToPython(value));
end;

procedure TNumPy.Setfloat96(const Value: TDtype);
begin
    FhModuleNumPy.SetAttr('float96', ToPython(value));
end;

procedure TNumPy.Setfloat_(const Value: TDtype);
begin
    FhModuleNumPy.SetAttr('float_', ToPython(value));
end;

procedure TNumPy.Sethalf(const Value: TDtype);
begin
    FhModuleNumPy.SetAttr('half', ToPython(value));
end;

procedure TNumPy.Setint16(const Value: TDtype);
begin
    FhModuleNumPy.SetAttr('int16', ToPython(value));
end;

procedure TNumPy.Setint32(const Value: TDtype);
begin
   FhModuleNumPy.SetAttr('int32', ToPython(value));
end;

procedure TNumPy.Setint64(const Value: TDtype);
begin
    FhModuleNumPy.SetAttr('int64', ToPython(value));
end;

procedure TNumPy.Setint8(const Value: TDtype);
begin
    FhModuleNumPy.SetAttr('int8', ToPython(value));
end;

procedure TNumPy.Setintc(const Value: TDtype);
begin
   FhModuleNumPy.SetAttr('intc', ToPython(value));
end;

procedure TNumPy.Setintp(const Value: TDtype);
begin
   FhModuleNumPy.SetAttr('intp', ToPython(value));
end;

procedure TNumPy.Setint_(const Value: TDtype);
begin
    FhModuleNumPy.SetAttr('int_', ToPython(value));
end;

procedure TNumPy.Setlongfloat(const Value: TDtype);
begin
    FhModuleNumPy.SetAttr('longfloat', ToPython(value));
end;

procedure TNumPy.Setlonglong(const Value: TDtype);
begin
    FhModuleNumPy.SetAttr('longlong', ToPython(value));
end;

procedure TNumPy.Setobject_(const Value: TDtype);
begin
   FhModuleNumPy.SetAttr('object_', ToPython(value));
end;

procedure TNumPy.Setshort(const Value: TDtype);
begin
   FhModuleNumPy.SetAttr('short', ToPython(value));
end;

procedure TNumPy.Setsingle(const Value: TDtype);
begin
    FhModuleNumPy.SetAttr('single', ToPython(value));
end;

procedure TNumPy.Setubyte(const Value: TDtype);
begin
   FhModuleNumPy.SetAttr('ubyte', ToPython(value));
end;

procedure TNumPy.Setuint(const Value: TDtype);
begin
   FhModuleNumPy.SetAttr('uint', ToPython(value));
end;

procedure TNumPy.Setuint16(const Value: TDtype);
begin
   FhModuleNumPy.SetAttr('uint16', ToPython(value));
end;

procedure TNumPy.Setuint32(const Value: TDtype);
begin
    FhModuleNumPy.SetAttr('uint32', ToPython(value));
end;

procedure TNumPy.Setuint64(const Value: TDtype);
begin
    FhModuleNumPy.SetAttr('uint64', ToPython(value));
end;

procedure TNumPy.Setuint8(const Value: TDtype);
begin
   FhModuleNumPy.SetAttr('uint8', ToPython(value));
end;

procedure TNumPy.Setuintc(const Value: TDtype);
begin
   FhModuleNumPy.SetAttr('uintc', ToPython(value));
end;

procedure TNumPy.Setuintp(const Value: TDtype);
begin
   FhModuleNumPy.SetAttr('uintp', ToPython(value));
end;

procedure TNumPy.Setulonglong(const Value: TDtype);
begin
    FhModuleNumPy.SetAttr('ulonglong', ToPython(value));
end;

procedure TNumPy.Setunicode_(const Value: TDtype);
begin
   FhModuleNumPy.SetAttr('unicode_', ToPython(value));
end;

procedure TNumPy.Setushort(const Value: TDtype);
begin
   FhModuleNumPy.SetAttr('ushort', ToPython(value));
end;

procedure TNumPy.Setvoid(const Value: TDtype);
begin
    FhModuleNumPy.SetAttr('void', ToPython(value));
end;

{ TNumPy_Constant }

/// <summary>
/// IEEE 754 floating point representation of (positive) infinity.
/// </summary>
function TNumPy.inf: Double;
begin
    Result := FhModuleNumPy.GetAttr('inf').AsDouble;
end;

/// <summary>
/// IEEE 754 floating point representation of (positive) infinity.
///
/// Use np.inf because Inf, Infinity, PINF and infty are aliases for inf.For more details, see inf.
/// </summary>
function TNumPy.Infinity: Double;
begin
    Result := FhModuleNumPy.GetAttr('inf').AsDouble;
end;

/// <summary>
/// IEEE 754 floating point representation of (positive) infinity.
///
/// Use np.inf because Inf, Infinity, PINF and infty are aliases for inf.For more details, see inf.
/// </summary>
function TNumPy.PINF: Double;
begin
   Result := FhModuleNumPy.GetAttr('inf').AsDouble;
end;

/// <summary>
/// IEEE 754 floating point representation of (positive) infinity.
///
/// Use np.inf because Inf, Infinity, PINF and infty are aliases for inf.For more details, see inf.
/// </summary>
function TNumPy.infty: Double;
begin
    Result := FhModuleNumPy.GetAttr('inf').AsDouble;
end;

/// <summary>
/// IEEE 754 floating point representation of (positive) infinity.
/// </summary>
function TNumPy.NINF: Double;
begin
   Result := FhModuleNumPy.GetAttr('NINF').AsDouble;
end;

/// <summary>
/// IEEE 754 floating point representation of Not a Number(NaN).
/// </summary>
function TNumPy.nan: Double;
begin
    Result := FhModuleNumPy.GetAttr('nan').AsDouble;
end;

/// <summary>
/// IEEE 754 floating point representation of negative zero.
/// </summary>
function TNumPy.NZERO: Double;
begin
   Result := FhModuleNumPy.GetAttr('NZERO').AsDouble;
end;

/// <summary>
/// IEEE 754 floating point representation of negative zero.
/// </summary>
function TNumPy.PZERO: Double;
begin
   Result := FhModuleNumPy.GetAttr('PZERO').AsDouble;
end;

/// <summary>
/// Euler’s constant, base of natural logarithms, Napier’s constant.
/// </summary>
function TNumPy.e: Double;
begin
   Result := FhModuleNumPy.GetAttr('e').AsDouble;
end;

/// <summary>
/// γ = 0.5772156649015328606065120900824024310421...
/// https://en.wikipedia.org/wiki/Euler-Mascheroni_constant
/// </summary>
function TNumPy.euler_gamma: Double;
begin
    Result := FhModuleNumPy.GetAttr('e').AsDouble;
end;

/// <summary>
/// A convenient alias for None, useful for indexing arrays.
/// </summary>
function TNumPy.newaxis: Double;
begin
    Result := FhModuleNumPy.GetAttr('newaxis').AsDouble;
end;

/// <summary>
/// pi = 3.1415926535897932384626433...
/// </summary>
function TNumPy.pi: Double;
begin
   Result := FhModuleNumPy.GetAttr('e').AsDouble;
end;

end.




