unit Keras.Layers;

interface
   uses System.Generics.Collections, System.Rtti, np.Utils,PythonEngine, Python.Utils, np.Models, Keras;

type
  TBaseLayer = class;

  //BaseLayer
  TBaseLayer = class(TBase)
     public
       constructor Create; overload;
       constructor Create(py: PPyObject); overload;
       constructor Create(py: TPythonObject); overload;
       function    &Set(inputs: TArray<TBaseLayer>): TBaseLayer;
  end;

  //Core

  TKInput = class(TBaseLayer)
     public
       constructor Create(shape       : Tnp_Shape ;
                          batch_shape : PTnp_Shape = nil;
                          name        : string = '';
                          dtype       : string = 'float32';
                          sparse      : Boolean= false;
                          tensor      : TNDarray= nil) ;
  end;

  TDense = class(TBaseLayer)
     public
       constructor Create(units               : Integer;
                          input_dim           : PInteger = nil;
                          activation          : string= '';
                          use_bias            : Boolean= true;
                          kernel_initializer  : string= 'glorot_uniform';
                          bias_initializer    : string= 'zeros';
                          kernel_regularizer  : string= '';
                          bias_regularizer    : string= '';
                          activity_regularizer: string= '';
                          kernel_constraint   : string= '';
                          bias_constraint     : string= '';
                          input_shape         : PTnp_Shape= nil) ; overload;

       constructor Create(units               : Integer;
                          activation          : string;
                          input_shape         : PTnp_Shape= nil) ;overload;

       constructor Create(units               : Integer;
                          activation          : string;
                          kernel_regularizer  : string;
                          input_shape         : PTnp_Shape) ;overload;
  end;

  TActivation = class(TBaseLayer)
     public
       constructor Create(act:string; input_shape : PTnp_Shape = nil) ;
  end;

  TDropout = class(TBaseLayer)
     public
       constructor Create(rate: Double; noise_shape: PTnp_Shape = nil; seed : PInteger = nil);
  end;

  TFlatten = class(TBaseLayer)
     public
       constructor Create(data_format: string = 'channels_last');
  end;

  TReshape = class(TBaseLayer)
     public
       constructor Create(target_shape: Tnp_Shape; input_shape: PTnp_Shape = nil);
  end;

  TPermute = class(TBaseLayer)
     public
       constructor Create(dims: Integer;input_shape : PTnp_Shape = nil);
  end;

  TRepeatVector = class(TBaseLayer)
     public
       constructor Create(n: Integer;input_shape : PTnp_Shape = nil);
  end;

  TLambda = class(TBaseLayer)
     public
       constructor Create(fun: PyCFunction; output_shape: PTnp_Shape = nil; mask: TNDarray = nil; arguments: TList< TPair<string,TValue> > = nil; input_shape: PTnp_Shape = nil);
  end;

  TActivityRegularization = class(TBaseLayer)
     public
       constructor Create(l1: double= 0.0; l2: Double = 0.0; input_shape: PTnp_Shape = nil);
  end;

  TMasking = class(TBaseLayer)
     public
       constructor Create(mask_value: Double = 0.0);
  end;

  TSpatialDropout1D = class(TBaseLayer)
     public
       constructor Create(rate: Double; input_shape: PTnp_Shape = nil);
  end;

  TSpatialDropout2D = class(TBaseLayer)
     public
       constructor Create(rate: Double; data_format: string = ''; input_shape: PTnp_Shape = nil);
  end;

  TSpatialDropout3D = class(TBaseLayer)
     public
       constructor Create(rate: Double; data_format: string = ''; input_shape: PTnp_Shape = nil);
  end;

  //AdvanceActivation

  TLeakyReLU = class(TBaseLayer)
     public
       constructor Create(alpha: Double = 0.3);
  end;

  TPReLU = class(TBaseLayer)
     public
       constructor Create(alpha_initializer: string= 'zeros'; alpha_regularizer: string = ''; alpha_constraint: string = ''; shared_axes: TArray<Integer>= nil);
  end;

  TELU = class(TBaseLayer)
     public
       constructor Create(alpha: Double = 1);
  end;

  TThresholdedReLU = class(TBaseLayer)
     public
       constructor Create(theta: Double = 1);
  end;

  TSoftmax = class(TBaseLayer)
     public
       constructor Create(axis: Integer = -1);
  end;

  TReLU = class(TBaseLayer)
     public
       constructor Create(max_value: PDouble= nil; negative_slope: Double= 0; threshold: Double= 0);
  end;

  //Convolution

  TConv1D = class(TBaseLayer)
     public
       constructor Create(filters             : Integer;
                          kernel_size         : Integer;
                          strides             : Integer= 1;
                          padding             : string= 'valid';
                          data_format         : string = 'channels_last';
                          dilation_rate       : Integer = 1;
                          activation          : string= '';
                          use_bias            : Boolean = true;
                          kernel_initializer  : string = 'glorot_uniform';
                          bias_initializer    : string = 'zeros';
                          kernel_regularizer  : string = '';
                          bias_regularizer    : string = '';
                          activity_regularizer: string = '';
                          kernel_constraint   : string = '';
                          bias_constraint     : string = '';
                          input_shape         : PTnp_Shape = nil); overload;

       constructor Create(filters             : Integer;
                          kernel_size         : Integer;
                          padding             : string;
                          activation          : string;
                          input_shape         : PTnp_Shape); overload;

       constructor Create(filters             : Integer;
                          kernel_size         : Integer;
                          padding             : string;
                          activation          : string;
                          input_shape         : PTnp_Shape;
                          kernel_regularizer  : string); overload;
  end;

  TConv2D = class(TBaseLayer)
     public
       constructor Create(filters             : Integer;
                          kernel_size         : TArray<Integer>;
                          strides             : TArray<Integer>= nil;
                          padding             : string= 'valid';
                          data_format         : string = 'channels_last';
                          dilation_rate       : TArray<Integer>= nil;
                          activation          : string= '';
                          use_bias            : Boolean = true;
                          kernel_initializer  : string = 'glorot_uniform';
                          bias_initializer    : string = 'zeros';
                          kernel_regularizer  : string = '';
                          bias_regularizer    : string = '';
                          activity_regularizer: string = '';
                          kernel_constraint   : string = '';
                          bias_constraint     : string = '';
                          input_shape         : PTnp_Shape = nil);overload;

       constructor Create(filters             : Integer;
                          kernel_size         : TArray<Integer>;
                          activation          : string= '');overload;

       constructor Create(filters    : Integer;
                          kernel_size: TArray<Integer>;
                          activation : string;
                          input_shape: PTnp_Shape); overload;
  end;

  TConv3D = class(TBaseLayer)
     public
       constructor Create(filters             : Integer;
                          kernel_size         : TArray<Integer>;
                          strides             : TArray<Integer>= nil;
                          padding             : string= 'valid';
                          data_format         : string = 'channels_last';
                          dilation_rate       : TArray<Integer>= nil;
                          activation          : string= '';
                          use_bias            : Boolean = true;
                          kernel_initializer  : string = 'glorot_uniform';
                          bias_initializer    : string = 'zeros';
                          kernel_regularizer  : string = '';
                          bias_regularizer    : string = '';
                          activity_regularizer: string = '';
                          kernel_constraint   : string = '';
                          bias_constraint     : string = '';
                          input_shape         : PTnp_Shape = nil);
  end;

  TSeparableConv1D = class(TBaseLayer)
     public
       constructor Create(filters               : Integer;
                          kernel_size           : Integer;
                          strides               : Integer= 1;
                          padding               : string= 'valid';
                          data_format           : string = 'channels_last';
                          dilation_rate         : Integer = 1;
                          depth_multiplier      : Integer = 1;
                          activation            : string= '';
                          use_bias              : Boolean = true;
                          depthwise_initializer : string = 'glorot_uniform';
                          pointwise_initializer : string = 'glorot_uniform';
                          bias_initializer      : string = 'zeros';
                          depthwise_regularizer : string = '';
                          pointwise_regularizer : string = '';
                          bias_regularizer      : string = '';
                          activity_regularizer  : string = '';
                          depthwise_constraint  : string = '';
                          pointwise_constraint  : string = '';
                          bias_constraint       : string = '';
                          input_shape           : PTnp_Shape = nil);
  end;

  TSeparableConv2D = class(TBaseLayer)
     public
       constructor Create(filters               : Integer;
                          kernel_size           : TArray<Integer>;
                          strides               : TArray<Integer> = nil;
                          padding               : string= 'valid';
                          data_format           : string = 'channels_last';
                          dilation_rate         : TArray<Integer> = nil;
                          depth_multiplier      : Integer = 1;
                          activation            : string= '';
                          use_bias              : Boolean = true;
                          depthwise_initializer : string = 'glorot_uniform';
                          pointwise_initializer : string = 'glorot_uniform';
                          bias_initializer      : string = 'zeros';
                          depthwise_regularizer : string = '';
                          pointwise_regularizer : string = '';
                          bias_regularizer      : string = '';
                          activity_regularizer  : string = '';
                          depthwise_constraint  : string = '';
                          pointwise_constraint  : string = '';
                          bias_constraint       : string = '';
                          input_shape           : PTnp_Shape = nil);
  end;

  TDepthwiseConv2D = class(TBaseLayer)
     public
       constructor Create(kernel_size           : TArray<Integer>;
                          strides               : TArray<Integer> = nil;
                          padding               : string= 'valid';
                          depth_multiplier      : Integer = 1;
                          data_format           : string = 'channels_last';
                          dilation_rate         : TArray<Integer> = nil;
                          activation            : string= '';
                          use_bias              : Boolean = true;
                          depthwise_initializer : string = 'glorot_uniform';
                          bias_initializer      : string = 'zeros';
                          depthwise_regularizer : string = '';
                          bias_regularizer      : string = '';
                          activity_regularizer  : string = '';
                          depthwise_constraint  : string = '';
                          bias_constraint       : string = '';
                          input_shape           : PTnp_Shape = nil);
  end;

  TConv2DTranspose = class(TBaseLayer)
     public
       constructor Create(filters               : Integer;
                          kernel_size           : TArray<Integer>;
                          strides               : TArray<Integer> = nil;
                          padding               : string= 'valid';
                          output_padding        : TArray<Integer> = nil;
                          data_format           : string = 'channels_last';
                          dilation_rate         : TArray<Integer> = nil;
                          activation            : string= '';
                          use_bias              : Boolean = true;
                          kernel_initializer    : string = 'glorot_uniform';
                          bias_initializer      : string = 'zeros';
                          kernel_regularizer    : string = '';
                          bias_regularizer      : string = '';
                          activity_regularizer  : string = '';
                          kernel_constraint     : string = '';
                          bias_constraint       : string = '';
                          input_shape           : PTnp_Shape = nil);
  end;

  TConv3DTranspose = class(TBaseLayer)
     public
       constructor Create(filters               : Integer;
                          kernel_size           : TArray<Integer>;
                          strides               : TArray<Integer> = nil;
                          padding               : string= 'valid';
                          output_padding        : TArray<Integer> = nil;
                          data_format           : string = 'channels_last';
                          dilation_rate         : TArray<Integer> = nil;
                          activation            : string= '';
                          use_bias              : Boolean = true;
                          kernel_initializer    : string = 'glorot_uniform';
                          bias_initializer      : string = 'zeros';
                          kernel_regularizer    : string = '';
                          bias_regularizer      : string = '';
                          activity_regularizer  : string = '';
                          kernel_constraint     : string = '';
                          bias_constraint       : string = '';
                          input_shape           : PTnp_Shape = nil);
  end;

  TCropping1D = class(TBaseLayer)
     public
       constructor Create(cropping : TArray<Integer>; input_shape : PTnp_Shape = nil);
  end;

  TCropping2D = class(TBaseLayer)
     public
       constructor Create(cropping : TArray<Tnp_Shape>;data_format : string = ''; input_shape : PTnp_Shape = nil);
  end;

  TCropping3D = class(TBaseLayer)
     public
       constructor Create(cropping : TArray<Tnp_Shape>; data_format : string = ''; input_shape : PTnp_Shape = nil);
  end;

  TUpSampling1D = class(TBaseLayer)
     public
       constructor Create(size: Integer = 2; input_shape : PTnp_Shape = nil);
  end;

  TUpSampling2D = class(TBaseLayer)
     public
       constructor Create(size: TArray<Integer>= nil; data_format   : string = ''; interpolation : string = 'nearest'; input_shape   : PTnp_Shape = nil);
  end;

  TUpSampling3D = class(TBaseLayer)
     public
       constructor Create(size: TArray<Integer>= nil; data_format   : string = ''; input_shape   : PTnp_Shape = nil);
  end;

  TZeroPadding1D = class(TBaseLayer)
     public
       constructor Create(padding: Integer= 1; input_shape   : PTnp_Shape = nil);
  end;

  TZeroPadding2D = class(TBaseLayer)
     public
       constructor Create(padding: TArray<Integer>= nil; data_format   : string = ''; input_shape   : PTnp_Shape = nil);
  end;

  TZeroPadding3D = class(TBaseLayer)
     public
       constructor Create(padding: TArray<Integer>= nil; data_format   : string = ''; input_shape   : PTnp_Shape = nil);
  end;

  //Embedding

  TEmbedding = class(TBaseLayer)
     public
       constructor Create(input_dim             : Integer;
                          output_dim            : Integer;
                          embeddings_initializer: string = 'uniform';
                          embeddings_regularizer: string = '';
                          activity_regularizer  : string = '';
                          embeddings_constraint : string = '';
                          mask_zero             : Boolean = false;
                          input_length          : PInteger = nil;
                          input_shape           : PTnp_Shape = nil); overload;

       constructor Create(input_dim             : Integer;
                          output_dim            : Integer;
                          input_length          : PInteger);overload;
  end;

  //LocallyConnected

  TLocallyConnected1D = class(TBaseLayer)
     public
       constructor Create(filters             : Integer;
                          kernel_size         : Integer;
                          strides             : Integer= 1;
                          padding             : string= 'valid';
                          data_format         : string = 'channels_last';
                          dilation_rate       : Integer = 1;
                          activation          : string= '';
                          use_bias            : Boolean = true;
                          kernel_initializer  : string = 'glorot_uniform';
                          bias_initializer    : string = 'zeros';
                          kernel_regularizer  : string = '';
                          bias_regularizer    : string = '';
                          activity_regularizer: string = '';
                          kernel_constraint   : string = '';
                          bias_constraint     : string = '';
                          input_shape         : PTnp_Shape = nil);
  end;

  TLocallyConnected2D = class(TBaseLayer)
     public
       constructor Create(filters             : Integer;
                          kernel_size         : TArray<Integer>;
                          strides             : TArray<Integer>= nil;
                          padding             : string= 'valid';
                          data_format         : string = 'channels_last';
                          dilation_rate       : TArray<Integer>= nil;
                          activation          : string= '';
                          use_bias            : Boolean = true;
                          kernel_initializer  : string = 'glorot_uniform';
                          bias_initializer    : string = 'zeros';
                          kernel_regularizer  : string = '';
                          bias_regularizer    : string = '';
                          activity_regularizer: string = '';
                          kernel_constraint   : string = '';
                          bias_constraint     : string = '';
                          input_shape         : PTnp_Shape = nil);
  end;

  //Merge

  TMerge = class(TBaseLayer)
     public
       constructor Create;
  end;

  TAdd = class(TMerge)
     public
       constructor Create(inputs : TArray<TBaseLayer>);
  end;

  TConcatenate = class(TMerge)
     public
       constructor Create(inputs : TArray<TBaseLayer>);
  end;

  //Noise

  TGaussianNoise = class(TBaseLayer)
     public
       constructor Create(stddev: Double);
  end;

  TGaussianDropout = class(TBaseLayer)
     public
       constructor Create(rate: Double);
  end;

  TAlphaDropout = class(TBaseLayer)
     public
       constructor Create(rate: Double; noise_shape: TNDarray<Integer> = nil; seed : PInteger= nil);
  end;

  //Normalization

  TBatchNormalization = class(TBaseLayer)
     public
       constructor Create(axis                        : Integer= -1;
                          momentum                    : Double= 0.99;
                          epsilon                     : Double =0.001;
                          center                      : Boolean= true;
                          scale                       : Boolean= true;
                          beta_initializer            : string = 'zeros';
                          gamma_initializer           : string = 'ones';
                          moving_mean_initializer     : string = 'zeros';
                          moving_variance_initializer : string = 'ones';
                          beta_regularizer            : string = '';
                          gamma_regularizer           : string = '';
                          beta_constraint             : string = '';
                          gamma_constraint            : string= '';
                          input_shape                 : PTnp_Shape = nil);
  end;

  // Pooling

  TMaxPooling1D = class(TBaseLayer)
     public
       constructor Create(pool_size: Integer = 2; strides: PInteger = nil; padding: string = 'valid'; data_format: string= 'channels_last');
  end;

  TMaxPooling2D = class(TBaseLayer)
     public
       constructor Create(pool_size: TArray<Integer>= nil ; strides: TArray<Integer> = nil; padding: string = 'valid'; data_format: string= 'channels_last');
  end;

  TMaxPooling3D = class(TBaseLayer)
     public
       constructor Create(pool_size: TArray<Integer>= nil ; strides: TArray<Integer> = nil; padding: string = 'valid'; data_format: string= 'channels_last');
  end;

  TAveragePooling1D = class(TBaseLayer)
     public
       constructor Create(pool_size: Integer = 2; strides: PInteger = nil; padding: string = 'valid'; data_format: string= 'channels_last');
  end;

  TAveragePooling2D = class(TBaseLayer)
     public
       constructor Create(pool_size: TArray<Integer>= nil ; strides: TArray<Integer> = nil; padding: string = 'valid'; data_format: string= 'channels_last');
  end;

  TAveragePooling3D = class(TBaseLayer)
     public
       constructor Create(pool_size: TArray<Integer>= nil ; strides: TArray<Integer> = nil; padding: string = 'valid'; data_format: string= 'channels_last');
  end;

  TGlobalMaxPooling1D = class(TBaseLayer)
     public
       constructor Create(data_format: string= 'channels_last');
  end;

  TGlobalMaxPooling2D = class(TBaseLayer)
     public
       constructor Create(data_format: string= 'channels_last');
  end;

  TGlobalMaxPooling3D = class(TBaseLayer)
     public
       constructor Create(data_format: string= 'channels_last');
  end;

  TGlobalAveragePooling1D = class(TBaseLayer)
     public
       constructor Create(data_format: string= 'channels_last');
  end;

  TGlobalAveragePooling2D = class(TBaseLayer)
     public
       constructor Create(data_format: string= 'channels_last');
  end;

  TGlobalAveragePooling3D = class(TBaseLayer)
     public
       constructor Create(data_format: string= 'channels_last');
  end;

  // Wrappers

  TTimeDistributed = class(TBaseLayer)
     public
       constructor Create(layer: TBaseLayer);
  end;

  TBidirectional = class(TBaseLayer)
     public
       constructor Create(layer: TBaseLayer;merge_mode: string= 'concat'; weights: TNDArray = nil; input_shape: PTnp_Shape= nil); overload;
       constructor Create(layer: TBaseLayer; input_shape: PTnp_Shape= nil); overload;
  end;

  //Recurrent

  TRNN = class(TBaseLayer)
     public
       constructor Create(cell             : TRNN;
                          return_sequences : Boolean = false;
                          return_state     : Boolean = false;
                          go_backwards     : Boolean = false;
                          stateful         : Boolean= false;
                          unroll           : Boolean= false;
                          input_dim        : PInteger = nil;
                          input_length     : PInteger= nil;
                          input_shape      : PTnp_Shape = nil);
  end;

  TSimpleRNN = class(TRNN)
     public
       constructor Create(units                : Integer;
                          activation           : string = 'tanh';
                          use_bias             : Boolean= true;
                          kernel_initializer   : string = 'glorot_uniform';
                          recurrent_initializer: string = 'orthogonal';
                          bias_initializer     : string = 'zeros';
                          kernel_regularizer   : string= '';
                          recurrent_regularizer: string= '';
                          bias_regularizer     : string= '';
                          activity_regularizer : string= '';
                          kernel_constraint    : string= '';
                          recurrent_constraint : string= '';
                          bias_constraint      : string= '';
                          dropout              : Double= 0.0;
                          recurrent_dropout    : Double= 0.0;
                          return_sequences     : Boolean= false;
                          return_state         : Boolean= false;
                          go_backwards         : Boolean= false;
                          stateful             : Boolean= false;
                          unroll               : Boolean= false);
  end;

  TSimpleRNNCell = class(TRNN)
     public
       constructor Create(units                : Integer;
                          activation           : string = 'tanh';
                          use_bias             : Boolean= true;
                          kernel_initializer   : string = 'glorot_uniform';
                          recurrent_initializer: string = 'orthogonal';
                          bias_initializer     : string = 'zeros';
                          kernel_regularizer   : string= '';
                          recurrent_regularizer: string= '';
                          bias_regularizer     : string= '';
                          activity_regularizer : string= '';
                          kernel_constraint    : string= '';
                          recurrent_constraint : string= '';
                          bias_constraint      : string= '';
                          dropout              : Double= 0.0;
                          recurrent_dropout    : Double= 0.0);
  end;

  TGRU = class(TRNN)
     public
       constructor Create(units                : Integer;
                          activation           : string = 'tanh';
                          recurrent_activation : string = 'sigmoid';
                          use_bias             : Boolean= true;
                          kernel_initializer   : string = 'glorot_uniform';
                          recurrent_initializer: string = 'orthogonal';
                          bias_initializer     : string = 'zeros';
                          kernel_regularizer   : string= '';
                          recurrent_regularizer: string= '';
                          bias_regularizer     : string= '';
                          activity_regularizer : string= '';
                          kernel_constraint    : string= '';
                          recurrent_constraint : string= '';
                          bias_constraint      : string= '';
                          dropout              : Double= 0.0;
                          recurrent_dropout    : Double= 0.0;
                          implement            : Integer = 1;
                          return_sequences     : Boolean= false;
                          return_state         : Boolean= false;
                          go_backwards         : Boolean= false;
                          stateful             : Boolean= false;
                          unroll               : Boolean= false;
                          reset_after          : Boolean= false);overload;

       constructor Create(units                : Integer;
                          dropout              : Double;
                          recurrent_dropout    : Double);overload;
  end;


  TCuDNNGRU = class(TRNN)
     public
       constructor Create(units                : Integer;
                          kernel_initializer   : string = 'glorot_uniform';
                          recurrent_initializer: string = 'orthogonal';
                          bias_initializer     : string = 'zeros';
                          kernel_regularizer   : string= '';
                          recurrent_regularizer: string= '';
                          bias_regularizer     : string= '';
                          activity_regularizer : string= '';
                          kernel_constraint    : string= '';
                          recurrent_constraint : string= '';
                          bias_constraint      : string= '';
                          return_sequences     : Boolean= false;
                          return_state         : Boolean= false;
                          stateful             : Boolean= false);
  end;

  TGRUCell = class(TRNN)
     public
       constructor Create(units                : Integer;
                          activation           : string = 'tanh';
                          recurrent_activation : string = 'sigmoid';
                          use_bias             : Boolean= true;
                          kernel_initializer   : string = 'glorot_uniform';
                          recurrent_initializer: string = 'orthogonal';
                          bias_initializer     : string = 'zeros';
                          kernel_regularizer   : string= '';
                          recurrent_regularizer: string= '';
                          bias_regularizer     : string= '';
                          kernel_constraint    : string= '';
                          recurrent_constraint : string= '';
                          bias_constraint      : string= '';
                          dropout              : Double= 0.0;
                          recurrent_dropout    : Double= 0.0;
                          implement            : Integer = 1;
                          reset_after          : Boolean= false);
  end;

  TLSTM = class(TRNN)
     public
       constructor Create(units                : Integer;
                          activation           : string = 'tanh';
                          recurrent_activation : string = 'sigmoid';
                          use_bias             : Boolean= true;
                          kernel_initializer   : string = 'glorot_uniform';
                          recurrent_initializer: string = 'orthogonal';
                          bias_initializer     : string = 'zeros';
                          unit_forget_bias     : Boolean =True;
                          kernel_regularizer   : string= '';
                          recurrent_regularizer: string= '';
                          bias_regularizer     : string= '';
                          activity_regularizer : string= '';
                          kernel_constraint    : string= '';
                          recurrent_constraint : string= '';
                          bias_constraint      : string= '';
                          dropout              : Double= 0.0;
                          recurrent_dropout    : Double= 0.0;
                          implement            : Integer = 1;
                          return_sequences     : Boolean= false;
                          return_state         : Boolean= false;
                          go_backwards         : Boolean= false;
                          stateful             : Boolean= false;
                          unroll               : Boolean= false;
                          input_shape          : PTnp_Shape= nil);overload;

       constructor Create(units                : Integer;
                          dropout              : Double;
                          recurrent_dropout    : Double);overload;

       constructor Create(units           : Integer;
                          activation      : string ;
                          input_shape     : PTnp_Shape;
                          stateful        : Boolean;
                          return_sequences: Boolean);  overload;
  end;

  TCuDNNLSTM = class(TRNN)
     public
       constructor Create(units                : Integer;
                          kernel_initializer   : string = 'glorot_uniform';
                          recurrent_initializer: string = 'orthogonal';
                          bias_initializer     : string = 'zeros';
                          unit_forget_bias     : Boolean =True;
                          kernel_regularizer   : string= '';
                          recurrent_regularizer: string= '';
                          bias_regularizer     : string= '';
                          activity_regularizer : string= '';
                          kernel_constraint    : string= '';
                          recurrent_constraint : string= '';
                          bias_constraint      : string= '';
                          return_sequences     : Boolean= false;
                          return_state         : Boolean= false;
                          stateful             : Boolean= false);
  end;

  TLSTMCell = class(TRNN)
     public
       constructor Create(units                : Integer;
                          activation           : string = 'tanh';
                          recurrent_activation : string = 'hard_sigmoid';
                          use_bias             : Boolean= true;
                          kernel_initializer   : string = 'glorot_uniform';
                          recurrent_initializer: string = 'orthogonal';
                          bias_initializer     : string = 'zeros';
                          unit_forget_bias     : Boolean =True;
                          kernel_regularizer   : string= '';
                          recurrent_regularizer: string= '';
                          bias_regularizer     : string= '';
                          kernel_constraint    : string= '';
                          recurrent_constraint : string= '';
                          bias_constraint      : string= '';
                          dropout              : Double= 0.0;
                          recurrent_dropout    : Double= 0.0;
                          implement            : Integer = 1);
  end;

  TConvLSTM2D = class(TBaseLayer)
     public
       constructor Create(filters              : Integer;
                          kernel_size          : TArray<Integer>;
                          strides              : TArray<Integer>= nil;
                          padding              : string= 'valid';
                          data_format          : string = '';
                          dilation_rate        : TArray<Integer>= nil;
                          activation           : string= 'tanh';
                          recurrent_activation : string = 'hard_sigmoid';
                          use_bias             : Boolean = true;
                          kernel_initializer   : string = 'glorot_uniform';
                          recurrent_initializer: string ='orthogonal';
                          bias_initializer     : string = 'zeros';
                          unit_forget_bias     : Boolean =True;
                          kernel_regularizer   : string = '';
                          recurrent_regularizer: string= '';
                          bias_regularizer     : string = '';
                          activity_regularizer : string = '';
                          kernel_constraint    : string = '';
                          recurrent_constraint : string= '';
                          bias_constraint      : string = '';
                          return_sequences     : Boolean= false;
                          go_backwards         : Boolean= false;
                          stateful             : Boolean= false;
                          dropout              : Double= 0.0;
                          recurrent_dropout    : Double= 0.0;
                          input_shape          : PTnp_Shape = nil);
  end;

  TConvLSTM2DCell = class(TBaseLayer)
     public
       constructor Create(filters              : Integer;
                          kernel_size          : TArray<Integer>;
                          strides              : TArray<Integer>= nil;
                          padding              : string= 'valid';
                          data_format          : string = '';
                          dilation_rate        : TArray<Integer>= nil;
                          activation           : string= 'tanh';
                          recurrent_activation : string = 'hard_sigmoid';
                          use_bias             : Boolean = true;
                          kernel_initializer   : string = 'glorot_uniform';
                          recurrent_initializer: string ='orthogonal';
                          bias_initializer     : string = 'zeros';
                          unit_forget_bias     : Boolean =True;
                          kernel_regularizer   : string = '';
                          recurrent_regularizer: string= '';
                          bias_regularizer     : string = '';
                          kernel_constraint    : string = '';
                          recurrent_constraint : string= '';
                          bias_constraint      : string = '';
                          dropout              : Double= 0.0;
                          recurrent_dropout    : Double= 0.0);
  end;


implementation


{ TBaseLayer }

constructor TBaseLayer.Create;
begin
    inherited Create;
end;

constructor TBaseLayer.Create(py: PPyObject);
begin
    Create;
    PyInstance := TPythonObject.Create( py );
end;

constructor TBaseLayer.Create(py: TPythonObject);
begin
    Create;
    PyInstance := py;
end;

function TBaseLayer.&Set(inputs: TArray<TBaseLayer>): TBaseLayer;
begin
    Parameters.Clear;

    if Length(inputs) = 1 then
    begin
        Parameters.Add( TPair<String,TValue>.Create('inputs', inputs[0].PyInstance) );

        Result := TBaseLayer.Create( InvokeMethod('__call__', Parameters) )
    end else
    begin
        var t : TPythonObject := inputs[0].PyInstance;
        var b : TBaseLayer    := TBaseLayer.Create(t);
        b.Init;
        Result := b;
    end;
end;

{ Input }

constructor TKInput.Create(shape:Tnp_Shape; batch_shape: PTnp_Shape; name, dtype: string; sparse: Boolean; tensor: TNDarray);
begin
    inherited Create;

    Parameters.Add( TPair<String,TValue>.Create('shape', TValue.FromShape(shape)));

    if batch_shape <> nil then Parameters.Add( TPair<String,TValue>.Create('batch_shape',TValue.FromShape(batch_shape^)))
    else                       Parameters.Add( TPair<String,TValue>.Create('batch_shape', TPythonObject.None ));

    Parameters.Add( TPair<String,TValue>.Create('name',name));
    Parameters.Add( TPair<String,TValue>.Create('dtype',dtype));
    Parameters.Add( TPair<String,TValue>.Create('sparse',sparse));
    Parameters.Add( TPair<String,TValue>.Create('tensor',tensor));

    PyInstance := GetKerasClassIstance('layers.Input');
    Init;
end;


{ TDense }

constructor TDense.Create(units: Integer; input_dim: PInteger; activation: string; use_bias: Boolean;
                                  kernel_initializer, bias_initializer, kernel_regularizer, bias_regularizer, activity_regularizer, kernel_constraint,
                                  bias_constraint: string; input_shape: PTnp_Shape);
begin
    inherited Create;

    Parameters.Add( TPair<String,TValue>.Create('units',units));

    if input_dim <> nil then Parameters.Add( TPair<String,TValue>.Create('input_dim',input_dim^))
    else                     Parameters.Add( TPair<String,TValue>.Create('input_dim', TPythonObject.None ));

    Parameters.Add( TPair<String,TValue>.Create('activation',activation));
    Parameters.Add( TPair<String,TValue>.Create('use_bias',use_bias));
    Parameters.Add( TPair<String,TValue>.Create('kernel_initializer',kernel_initializer));
    Parameters.Add( TPair<String,TValue>.Create('bias_initializer',bias_initializer));
    Parameters.Add( TPair<String,TValue>.Create('kernel_regularizer',kernel_regularizer));
    Parameters.Add( TPair<String,TValue>.Create('bias_regularizer',bias_regularizer));
    Parameters.Add( TPair<String,TValue>.Create('activity_regularizer',activity_regularizer));
    Parameters.Add( TPair<String,TValue>.Create('kernel_constraint',kernel_constraint));
    Parameters.Add( TPair<String,TValue>.Create('bias_constraint',bias_constraint));

    if input_shape <> nil then Parameters.Add( TPair<String,TValue>.Create('input_shape',TValue.FromShape(input_shape^)))
    else                       Parameters.Add( TPair<String,TValue>.Create('input_shape', TPythonObject.None ));

    PyInstance := GetKerasClassIstance('layers.Dense');
    Init;
end;

constructor TDense.Create(units: Integer; activation: string; input_shape: PTnp_Shape);
begin
    inherited Create;

    Parameters.Add( TPair<String,TValue>.Create('units',units));
    Parameters.Add( TPair<String,TValue>.Create('activation',activation));

    if input_shape <> nil then Parameters.Add( TPair<String,TValue>.Create('input_shape',TValue.FromShape(input_shape^)))
    else                       Parameters.Add( TPair<String,TValue>.Create('input_shape', TPythonObject.None ));

    PyInstance := GetKerasClassIstance('layers.Dense');
    Init;
end;

constructor TDense.Create(units : Integer; activation : string; kernel_regularizer: string; input_shape: PTnp_Shape) ;
begin
    inherited Create;

    Parameters.Add( TPair<String,TValue>.Create('units',units));
    Parameters.Add( TPair<String,TValue>.Create('activation',activation));
    Parameters.Add( TPair<String,TValue>.Create('kernel_regularizer',kernel_regularizer));

    if input_shape <> nil then Parameters.Add( TPair<String,TValue>.Create('input_shape',TValue.FromShape(input_shape^)))
    else                       Parameters.Add( TPair<String,TValue>.Create('input_shape', TPythonObject.None ));

    PyInstance := GetKerasClassIstance('layers.Dense');
    Init;
end;

{ TActivation }

constructor TActivation.Create(act: string; input_shape: PTnp_Shape);
begin
    inherited Create;

    Parameters.Add( TPair<String,TValue>.Create('activation',act));

    if input_shape <> nil then Parameters.Add( TPair<String,TValue>.Create('input_shape',TValue.FromShape(input_shape^)))
    else                       Parameters.Add( TPair<String,TValue>.Create('input_shape', TPythonObject.None ));

    PyInstance := GetKerasClassIstance('layers.Activation');
    Init;
end;

{ TDropout }

constructor TDropout.Create(rate: Double; noise_shape: PTnp_Shape; seed: PInteger);
begin
    inherited Create;

    Parameters.Add( TPair<String,TValue>.Create('rate',rate));

    if noise_shape <> nil then Parameters.Add( TPair<String,TValue>.Create('noise_shape',TValue.FromShape(noise_shape^)))
    else                       Parameters.Add( TPair<String,TValue>.Create('noise_shape', TPythonObject.None ));

    if seed <> nil then Parameters.Add( TPair<String,TValue>.Create('seed',seed^))
    else                Parameters.Add( TPair<String,TValue>.Create('seed', TPythonObject.None ));

    PyInstance := GetKerasClassIstance('layers.Dropout');
    Init;
end;

{ TFlatten }

constructor TFlatten.Create(data_format: string);
begin
    inherited Create;

    Parameters.Add( TPair<String,TValue>.Create('data_format',data_format));

    PyInstance := GetKerasClassIstance('layers.Flatten');
    Init;
end;

{ TReshape }

constructor TReshape.Create(target_shape: Tnp_Shape; input_shape: PTnp_Shape);
begin
    inherited Create;

    Parameters.Add( TPair<String,TValue>.Create('target_shape', TValue.FromShape(target_shape)));

    if input_shape <> nil then Parameters.Add( TPair<String,TValue>.Create('input_shape',TValue.FromShape(input_shape^)))
    else                       Parameters.Add( TPair<String,TValue>.Create('input_shape', TPythonObject.None ));

    PyInstance := GetKerasClassIstance('layers.Reshape');
    Init;
end;

{ TPermute }

constructor TPermute.Create(dims: Integer; input_shape: PTnp_Shape);
begin
    inherited Create;

    Parameters.Add( TPair<String,TValue>.Create('dims', dims));

    if input_shape <> nil then Parameters.Add( TPair<String,TValue>.Create('input_shape',TValue.FromShape(input_shape^)))
    else                       Parameters.Add( TPair<String,TValue>.Create('input_shape', TPythonObject.None ));

    PyInstance := GetKerasClassIstance('layers.Permute');
    Init;
end;

{ TRepeatVector }

constructor TRepeatVector.Create(n: Integer; input_shape: PTnp_Shape);
begin
    inherited Create;

    Parameters.Add( TPair<String,TValue>.Create('n', n));

    if input_shape <> nil then Parameters.Add( TPair<String,TValue>.Create('input_shape',TValue.FromShape(input_shape^)))
    else                       Parameters.Add( TPair<String,TValue>.Create('input_shape', TPythonObject.None ));

    PyInstance := GetKerasClassIstance('layers.RepeatVector');
    Init;
end;

{ TLambda }

constructor TLambda.Create(fun: PyCFunction; output_shape: PTnp_Shape; mask: TNDarray;
                                 arguments: TList<TPair<string, TValue>>; input_shape: PTnp_Shape);
begin
    inherited Create;

    var m : TPythonModule := TPythonModule.Create(nil);
    m.ModuleName := 'kcallbacks';
    m.Engine     := g_MyPyEngine;
    m.Initialize;

    var mr : PPyMethodDef := m.AddMethod(pansichar('cbLambda'),fun,PAnsiChar('Lambda Callback Func.'));
    CreatePyFunc(m,mr);

    var cb : TPythonObject := TPythonObject.create(m.Module).GetAttr('cbLambda') ;

    Parameters.Add( TPair<String,TValue>.Create('function', cb));

    if input_shape <> nil then Parameters.Add( TPair<String,TValue>.Create('output_shape',TValue.FromShape(output_shape^)))
    else                       Parameters.Add( TPair<String,TValue>.Create('output_shape', TPythonObject.None ));

    Parameters.Add( TPair<String,TValue>.Create('mask', mask));

    Parameters.Add( TPair<String,TValue>.Create('arguments', arguments));

    if input_shape <> nil then Parameters.Add( TPair<String,TValue>.Create('input_shape',TValue.FromShape(input_shape^)))
    else                       Parameters.Add( TPair<String,TValue>.Create('input_shape', TPythonObject.None ));

    PyInstance := GetKerasClassIstance('layers.Lambda');
    Init;
end;

{ TActivityRegularization }

constructor TActivityRegularization.Create(l1, l2: Double; input_shape: PTnp_Shape);
begin
    inherited Create;

    Parameters.Add( TPair<String,TValue>.Create('l1', l1));
    Parameters.Add( TPair<String,TValue>.Create('l2', l2));

    if input_shape <> nil then Parameters.Add( TPair<String,TValue>.Create('input_shape',TValue.FromShape(input_shape^)))
    else                       Parameters.Add( TPair<String,TValue>.Create('input_shape', TPythonObject.None ));

    PyInstance := GetKerasClassIstance('layers.ActivityRegularization');
    Init;
end;

{ TMasking }

constructor TMasking.Create(mask_value: Double);
begin
    inherited Create;

    Parameters.Add( TPair<String,TValue>.Create('mask_value', mask_value));

    PyInstance := GetKerasClassIstance('layers.Masking');
    Init;
end;

{ TSpatialDropout1D }

constructor TSpatialDropout1D.Create(rate: Double; input_shape: PTnp_Shape);
begin
    inherited Create;

    Parameters.Add( TPair<String,TValue>.Create('rate', rate));

    if input_shape <> nil then Parameters.Add( TPair<String,TValue>.Create('input_shape',TValue.FromShape(input_shape^)))
    else                       Parameters.Add( TPair<String,TValue>.Create('input_shape', TPythonObject.None ));

    PyInstance := GetKerasClassIstance('layers.SpatialDropout1D');
    Init;
end;

{ TSpatialDropout2D }

constructor TSpatialDropout2D.Create(rate: Double; data_format: string; input_shape: PTnp_Shape);
begin
    inherited Create;

    Parameters.Add( TPair<String,TValue>.Create('rate', rate));
    Parameters.Add( TPair<String,TValue>.Create('data_format', data_format));

    if input_shape <> nil then Parameters.Add( TPair<String,TValue>.Create('input_shape',TValue.FromShape(input_shape^)))
    else                       Parameters.Add( TPair<String,TValue>.Create('input_shape', TPythonObject.None ));

    PyInstance := GetKerasClassIstance('layers.SpatialDropout2D');
    Init;
end;

{ TSpatialDropout3D }

constructor TSpatialDropout3D.Create(rate: Double; data_format: string; input_shape: PTnp_Shape);
begin
    inherited Create;

    Parameters.Add( TPair<String,TValue>.Create('rate', rate));
    Parameters.Add( TPair<String,TValue>.Create('data_format', data_format));

    if input_shape <> nil then Parameters.Add( TPair<String,TValue>.Create('input_shape',TValue.FromShape(input_shape^)))
    else                       Parameters.Add( TPair<String,TValue>.Create('input_shape', TPythonObject.None ));

    PyInstance := GetKerasClassIstance('layers.SpatialDropout3D');
    Init;
end;

{ TLeakyReLU }

constructor TLeakyReLU.Create(alpha: Double);
begin
    inherited Create;

    Parameters.Add( TPair<String,TValue>.Create('alpha', alpha));

    PyInstance := GetKerasClassIstance('layers.LeakyReLU');
    Init;
end;

{ TPReLU }

constructor TPReLU.Create(alpha_initializer, alpha_regularizer, alpha_constraint: string; shared_axes: TArray<Integer>);
begin
    inherited Create;

    Parameters.Add( TPair<String,TValue>.Create('alpha_initializer', alpha_initializer));
    Parameters.Add( TPair<String,TValue>.Create('alpha_regularizer', alpha_regularizer));
    Parameters.Add( TPair<String,TValue>.Create('alpha_constraint', alpha_constraint));

    if shared_axes <> nil then Parameters.Add( TPair<String,TValue>.Create('shared_axes',TValue.FromArray<Integer>(shared_axes)))
    else                       Parameters.Add( TPair<String,TValue>.Create('shared_axes', TPythonObject.None ));

    PyInstance := GetKerasClassIstance('layers.PReLU');
    Init;
end;

{ TELU }

constructor TELU.Create(alpha: Double);
begin
    inherited Create;

    Parameters.Add( TPair<String,TValue>.Create('alpha', alpha));

    PyInstance := GetKerasClassIstance('layers.ELU');
    Init;
end;

{ TThresholdedReLU }

constructor TThresholdedReLU.Create(theta: Double);
begin
    inherited Create;

    Parameters.Add( TPair<String,TValue>.Create('theta', theta));

    PyInstance := GetKerasClassIstance('layers.ThresholdedReLU');
    Init;
end;

{ TSoftmax }

constructor TSoftmax.Create(axis: Integer);
begin
    inherited Create;

    Parameters.Add( TPair<String,TValue>.Create('axis', axis));

    PyInstance := GetKerasClassIstance('layers.Softmax');
    Init;
end;

{ TReLU }

constructor TReLU.Create(max_value:PDouble; negative_slope: Double; threshold: Double);
begin
    inherited Create;

    if max_value <> nil then Parameters.Add( TPair<String,TValue>.Create('max_value',max_value^))
    else                     Parameters.Add( TPair<String,TValue>.Create('max_value', TPythonObject.None ));

    Parameters.Add( TPair<String,TValue>.Create('negative_slope', negative_slope));
    Parameters.Add( TPair<String,TValue>.Create('threshold', threshold));


    PyInstance := GetKerasClassIstance('layers.ReLU');
    Init;
end;

{ TConv1D }

constructor TConv1D.Create(filters, kernel_size, strides: Integer; padding, data_format: string; dilation_rate: Integer;
                            activation: string; use_bias: Boolean; kernel_initializer, bias_initializer, kernel_regularizer, bias_regularizer,
                            activity_regularizer, kernel_constraint, bias_constraint: string; input_shape: PTnp_Shape);
begin
    inherited Create;

    Parameters.Add( TPair<String,TValue>.Create('filters',filters));
    Parameters.Add( TPair<String,TValue>.Create('kernel_size',kernel_size));
    Parameters.Add( TPair<String,TValue>.Create('strides',strides));
    Parameters.Add( TPair<String,TValue>.Create('padding',padding));
    Parameters.Add( TPair<String,TValue>.Create('data_format',data_format));
    Parameters.Add( TPair<String,TValue>.Create('dilation_rate',dilation_rate));
    Parameters.Add( TPair<String,TValue>.Create('activation',activation));
    Parameters.Add( TPair<String,TValue>.Create('use_bias',use_bias));
    Parameters.Add( TPair<String,TValue>.Create('kernel_initializer',kernel_initializer));
    Parameters.Add( TPair<String,TValue>.Create('bias_initializer',bias_initializer));
    Parameters.Add( TPair<String,TValue>.Create('kernel_regularizer',kernel_regularizer));
    Parameters.Add( TPair<String,TValue>.Create('bias_regularizer',bias_regularizer));
    Parameters.Add( TPair<String,TValue>.Create('activity_regularizer',activity_regularizer));
    Parameters.Add( TPair<String,TValue>.Create('kernel_constraint',kernel_constraint));
    Parameters.Add( TPair<String,TValue>.Create('bias_constraint',bias_constraint));

    if input_shape <> nil then Parameters.Add( TPair<String,TValue>.Create('input_shape',TValue.FromShape(input_shape^)))
    else                       Parameters.Add( TPair<String,TValue>.Create('input_shape', TPythonObject.None ));

    PyInstance := GetKerasClassIstance('layers.Conv1D');
    Init;
end;

constructor TConv1D.Create(filters, kernel_size: Integer; padding, activation: string; input_shape: PTnp_Shape);
begin
    inherited Create;

    Parameters.Add( TPair<String,TValue>.Create('filters',filters));
    Parameters.Add( TPair<String,TValue>.Create('kernel_size',kernel_size));
    Parameters.Add( TPair<String,TValue>.Create('padding',padding));
    Parameters.Add( TPair<String,TValue>.Create('activation',activation));

    if input_shape <> nil then Parameters.Add( TPair<String,TValue>.Create('input_shape',TValue.FromShape(input_shape^)))
    else                       Parameters.Add( TPair<String,TValue>.Create('input_shape', TPythonObject.None ));

    PyInstance := GetKerasClassIstance('layers.Conv1D');
    Init;
end;

constructor TConv1D.Create(filters, kernel_size: Integer; padding, activation: string; input_shape: PTnp_Shape;kernel_regularizer : string);
begin
    inherited Create;

    Parameters.Add( TPair<String,TValue>.Create('filters',filters));
    Parameters.Add( TPair<String,TValue>.Create('kernel_size',kernel_size));
    Parameters.Add( TPair<String,TValue>.Create('padding',padding));
    Parameters.Add( TPair<String,TValue>.Create('activation',activation));
    Parameters.Add( TPair<String,TValue>.Create('kernel_regularizer',kernel_regularizer));

    if input_shape <> nil then Parameters.Add( TPair<String,TValue>.Create('input_shape',TValue.FromShape(input_shape^)))
    else                       Parameters.Add( TPair<String,TValue>.Create('input_shape', TPythonObject.None ));

    PyInstance := GetKerasClassIstance('layers.Conv1D');
    Init;
end;

{ TConv2D }

constructor TConv2D.Create(filters: Integer; kernel_size, strides: TArray<Integer>; padding, data_format: string;
                  dilation_rate: TArray<Integer>; activation: string; use_bias: Boolean; kernel_initializer, bias_initializer,
                  kernel_regularizer, bias_regularizer, activity_regularizer, kernel_constraint, bias_constraint: string;
                  input_shape: PTnp_Shape);
var
  ksize, stride,drate : TValue;
begin
    inherited Create;

    ksize := TValue.FromShape( Tnp_Shape.Create([kernel_size[0], kernel_size[1]]) );

    if strides = nil then stride := TValue.FromShape( Tnp_Shape.Create([1, 1]) )
    else                  stride := TValue.FromShape( Tnp_Shape.Create([strides[0], strides[1]]) );

    if dilation_rate = nil then drate := TValue.FromShape( Tnp_Shape.Create([1, 1]) )
    else                        drate := TValue.FromShape( Tnp_Shape.Create([dilation_rate[0], dilation_rate[1]]) );


    Parameters.Add( TPair<String,TValue>.Create('filters',filters));
    Parameters.Add( TPair<String,TValue>.Create('kernel_size', ksize ));
    Parameters.Add( TPair<String,TValue>.Create('strides',stride ));
    Parameters.Add( TPair<String,TValue>.Create('padding',padding));
    Parameters.Add( TPair<String,TValue>.Create('data_format',data_format));
    Parameters.Add( TPair<String,TValue>.Create('dilation_rate',drate));
    Parameters.Add( TPair<String,TValue>.Create('activation',activation));
    Parameters.Add( TPair<String,TValue>.Create('use_bias',use_bias));
    Parameters.Add( TPair<String,TValue>.Create('kernel_initializer',kernel_initializer));
    Parameters.Add( TPair<String,TValue>.Create('bias_initializer',bias_initializer));
    Parameters.Add( TPair<String,TValue>.Create('kernel_regularizer',kernel_regularizer));
    Parameters.Add( TPair<String,TValue>.Create('bias_regularizer',bias_regularizer));
    Parameters.Add( TPair<String,TValue>.Create('activity_regularizer',activity_regularizer));
    Parameters.Add( TPair<String,TValue>.Create('kernel_constraint',kernel_constraint));
    Parameters.Add( TPair<String,TValue>.Create('bias_constraint',bias_constraint));

    if input_shape <> nil then Parameters.Add( TPair<String,TValue>.Create('input_shape',TValue.FromShape(input_shape^)))
    else                       Parameters.Add( TPair<String,TValue>.Create('input_shape', TPythonObject.None ));


    PyInstance := GetKerasClassIstance('layers.Conv2D');
    Init;
end;

constructor TConv2D.Create(filters: Integer; kernel_size: TArray<Integer>; activation: string; input_shape: PTnp_Shape);
var
  ksize : TValue;
begin
    inherited Create;

    ksize := TValue.FromShape( Tnp_Shape.Create([kernel_size[0], kernel_size[1]]) );

    Parameters.Add( TPair<String,TValue>.Create('filters',filters));
    Parameters.Add( TPair<String,TValue>.Create('kernel_size', ksize ));
    Parameters.Add( TPair<String,TValue>.Create('activation',activation));

    if input_shape <> nil then Parameters.Add( TPair<String,TValue>.Create('input_shape',TValue.FromShape(input_shape^)))
    else                       Parameters.Add( TPair<String,TValue>.Create('input_shape', TPythonObject.None ));


    PyInstance := GetKerasClassIstance('layers.Conv2D');
    Init;

end;


constructor TConv2D.Create(filters: Integer; kernel_size: TArray<Integer>; activation: string);
var
  ksize : TValue;
begin
    inherited Create;

    ksize := TValue.FromShape( Tnp_Shape.Create([kernel_size[0], kernel_size[1]]) );

    Parameters.Add( TPair<String,TValue>.Create('filters',filters));
    Parameters.Add( TPair<String,TValue>.Create('kernel_size', ksize ));
    Parameters.Add( TPair<String,TValue>.Create('activation',activation));

    PyInstance := GetKerasClassIstance('layers.Conv2D');
    Init;

end;

{ TConv3D }

constructor TConv3D.Create(filters: Integer; kernel_size, strides: TArray<Integer>; padding, data_format: string;
                              dilation_rate: TArray<Integer>; activation: string; use_bias: Boolean; kernel_initializer, bias_initializer,
                              kernel_regularizer, bias_regularizer, activity_regularizer, kernel_constraint, bias_constraint: string;
                              input_shape: PTnp_Shape);
var
  ksize, stride,drate : TValue;
begin
    inherited Create;

    ksize := TValue.FromShape( Tnp_Shape.Create([kernel_size[0], kernel_size[1],kernel_size[2]]) );

    if strides = nil then stride := TValue.FromShape( Tnp_Shape.Create([1, 1, 1]) )
    else                  stride := TValue.FromShape( Tnp_Shape.Create([strides[0], strides[1], strides[2]]) );

    if dilation_rate = nil then drate := TValue.FromShape( Tnp_Shape.Create([1, 1, 1]) )
    else                        drate := TValue.FromShape( Tnp_Shape.Create([dilation_rate[0], dilation_rate[1], dilation_rate[2]]) );

    Parameters.Add( TPair<String,TValue>.Create('filters',filters));
    Parameters.Add( TPair<String,TValue>.Create('kernel_size',ksize));
    Parameters.Add( TPair<String,TValue>.Create('strides',stride));
    Parameters.Add( TPair<String,TValue>.Create('padding',padding));
    Parameters.Add( TPair<String,TValue>.Create('data_format',data_format));
    Parameters.Add( TPair<String,TValue>.Create('dilation_rate',drate ));
    Parameters.Add( TPair<String,TValue>.Create('activation',activation));
    Parameters.Add( TPair<String,TValue>.Create('use_bias',use_bias));
    Parameters.Add( TPair<String,TValue>.Create('kernel_initializer',kernel_initializer));
    Parameters.Add( TPair<String,TValue>.Create('bias_initializer',bias_initializer));
    Parameters.Add( TPair<String,TValue>.Create('kernel_regularizer',kernel_regularizer));
    Parameters.Add( TPair<String,TValue>.Create('bias_regularizer',bias_regularizer));
    Parameters.Add( TPair<String,TValue>.Create('activity_regularizer',activity_regularizer));
    Parameters.Add( TPair<String,TValue>.Create('kernel_constraint',kernel_constraint));
    Parameters.Add( TPair<String,TValue>.Create('bias_constraint',bias_constraint));

    if input_shape <> nil then Parameters.Add( TPair<String,TValue>.Create('input_shape',TValue.FromShape(input_shape^)))
    else                       Parameters.Add( TPair<String,TValue>.Create('input_shape', TPythonObject.None ));

    PyInstance := GetKerasClassIstance('layers.Conv3D');
    Init;
end;

{ TSeparableConv1D }

constructor TSeparableConv1D.Create(filters, kernel_size, strides: Integer; padding, data_format: string; dilation_rate,
                    depth_multiplier: Integer; activation: string; use_bias: Boolean; depthwise_initializer, pointwise_initializer,
                    bias_initializer, depthwise_regularizer,pointwise_regularizer, bias_regularizer, activity_regularizer, depthwise_constraint,
                    pointwise_constraint, bias_constraint: string; input_shape: PTnp_Shape);
begin
    inherited Create;

    Parameters.Add( TPair<String,TValue>.Create('filters',filters));
    Parameters.Add( TPair<String,TValue>.Create('kernel_size',kernel_size));
    Parameters.Add( TPair<String,TValue>.Create('strides',strides));
    Parameters.Add( TPair<String,TValue>.Create('padding',padding));
    Parameters.Add( TPair<String,TValue>.Create('data_format',data_format));
    Parameters.Add( TPair<String,TValue>.Create('dilation_rate',dilation_rate));
    Parameters.Add( TPair<String,TValue>.Create('depth_multiplier',depth_multiplier));
    Parameters.Add( TPair<String,TValue>.Create('activation',activation));
    Parameters.Add( TPair<String,TValue>.Create('use_bias',use_bias));
    Parameters.Add( TPair<String,TValue>.Create('depthwise_initializer',depthwise_initializer));
    Parameters.Add( TPair<String,TValue>.Create('pointwise_initializer',pointwise_initializer));
    Parameters.Add( TPair<String,TValue>.Create('bias_initializer',bias_initializer));
    Parameters.Add( TPair<String,TValue>.Create('depthwise_regularizer',depthwise_regularizer));
    Parameters.Add( TPair<String,TValue>.Create('bias_initializer',bias_initializer));
    Parameters.Add( TPair<String,TValue>.Create('pointwise_regularizer',pointwise_regularizer));
    Parameters.Add( TPair<String,TValue>.Create('bias_regularizer',bias_regularizer));
    Parameters.Add( TPair<String,TValue>.Create('activity_regularizer',activity_regularizer));
    Parameters.Add( TPair<String,TValue>.Create('depthwise_constraint',depthwise_constraint));
    Parameters.Add( TPair<String,TValue>.Create('pointwise_constraint',pointwise_constraint));
    Parameters.Add( TPair<String,TValue>.Create('bias_constraint',bias_constraint));

    if input_shape <> nil then Parameters.Add( TPair<String,TValue>.Create('input_shape',TValue.FromShape(input_shape^)))
    else                       Parameters.Add( TPair<String,TValue>.Create('input_shape', TPythonObject.None ));

    PyInstance := GetKerasClassIstance('layers.SeparableConv1D');
    Init;
end;

{ TSeparableConv2D }

constructor TSeparableConv2D.Create(filters: Integer; kernel_size, strides: TArray<Integer>; padding,
                                    data_format: string; dilation_rate: TArray<Integer>; depth_multiplier: Integer; activation: string; use_bias: Boolean;
                                    depthwise_initializer, pointwise_initializer, bias_initializer, depthwise_regularizer, pointwise_regularizer, bias_regularizer,
                                    activity_regularizer, depthwise_constraint, pointwise_constraint, bias_constraint: string; input_shape: PTnp_Shape);
var
  ksize, stride,drate : TValue;
begin
    inherited Create;

    ksize := TValue.FromShape( Tnp_Shape.Create([kernel_size[0], kernel_size[1]]) );

    if strides = nil then stride := TValue.FromShape( Tnp_Shape.Create([1, 1]) )
    else                  stride := TValue.FromShape( Tnp_Shape.Create([strides[0], strides[1]]) );

    if dilation_rate = nil then drate := TValue.FromShape( Tnp_Shape.Create([1, 1]) )
    else                        drate := TValue.FromShape( Tnp_Shape.Create([dilation_rate[0], dilation_rate[1]]) );

    Parameters.Add( TPair<String,TValue>.Create('filters',filters));
    Parameters.Add( TPair<String,TValue>.Create('kernel_size',ksize));
    Parameters.Add( TPair<String,TValue>.Create('strides',stride));
    Parameters.Add( TPair<String,TValue>.Create('padding',padding));
    Parameters.Add( TPair<String,TValue>.Create('data_format',data_format));
    Parameters.Add( TPair<String,TValue>.Create('dilation_rate',drate)) ;
    Parameters.Add( TPair<String,TValue>.Create('depth_multiplier',depth_multiplier));
    Parameters.Add( TPair<String,TValue>.Create('activation',activation));
    Parameters.Add( TPair<String,TValue>.Create('use_bias',use_bias));
    Parameters.Add( TPair<String,TValue>.Create('depthwise_initializer',depthwise_initializer));
    Parameters.Add( TPair<String,TValue>.Create('pointwise_initializer',pointwise_initializer));
    Parameters.Add( TPair<String,TValue>.Create('bias_initializer',bias_initializer));
    Parameters.Add( TPair<String,TValue>.Create('depthwise_regularizer',depthwise_regularizer));
    Parameters.Add( TPair<String,TValue>.Create('bias_initializer',bias_initializer));
    Parameters.Add( TPair<String,TValue>.Create('pointwise_regularizer',pointwise_regularizer));
    Parameters.Add( TPair<String,TValue>.Create('bias_regularizer',bias_regularizer));
    Parameters.Add( TPair<String,TValue>.Create('activity_regularizer',activity_regularizer));
    Parameters.Add( TPair<String,TValue>.Create('depthwise_constraint',depthwise_constraint));
    Parameters.Add( TPair<String,TValue>.Create('pointwise_constraint',pointwise_constraint));
    Parameters.Add( TPair<String,TValue>.Create('bias_constraint',bias_constraint));

    if input_shape <> nil then Parameters.Add( TPair<String,TValue>.Create('input_shape',TValue.FromShape(input_shape^)))
    else                       Parameters.Add( TPair<String,TValue>.Create('input_shape', TPythonObject.None ));

    PyInstance := GetKerasClassIstance('layers.SeparableConv2D');
    Init;
end;

{ TDepthwiseConv2D }

constructor TDepthwiseConv2D.Create(kernel_size, strides: TArray<Integer>; padding: string; depth_multiplier: Integer;
                      data_format: string; dilation_rate: TArray<Integer>; activation: string; use_bias: Boolean; depthwise_initializer,
                      bias_initializer, depthwise_regularizer, bias_regularizer, activity_regularizer, depthwise_constraint,
                      bias_constraint: string; input_shape: PTnp_Shape);
var
  ksize, stride : TValue;
begin
    inherited Create;

    ksize := TValue.FromShape( Tnp_Shape.Create([kernel_size[0], kernel_size[1]]) );

    if strides = nil then stride := TValue.FromShape( Tnp_Shape.Create([1, 1]) )
    else                  stride := TValue.FromShape( Tnp_Shape.Create([strides[0], strides[1]]) );

    Parameters.Add( TPair<String,TValue>.Create('kernel_size',ksize));
    Parameters.Add( TPair<String,TValue>.Create('strides',stride));
    Parameters.Add( TPair<String,TValue>.Create('padding',padding));
    Parameters.Add( TPair<String,TValue>.Create('depth_multiplier',depth_multiplier));
    Parameters.Add( TPair<String,TValue>.Create('data_format',data_format));
    Parameters.Add( TPair<String,TValue>.Create('activation',activation));
    Parameters.Add( TPair<String,TValue>.Create('use_bias',use_bias));
    Parameters.Add( TPair<String,TValue>.Create('depthwise_initializer',depthwise_initializer));
    Parameters.Add( TPair<String,TValue>.Create('bias_initializer',bias_initializer));
    Parameters.Add( TPair<String,TValue>.Create('depthwise_regularizer',depthwise_regularizer));
    Parameters.Add( TPair<String,TValue>.Create('bias_initializer',bias_initializer));
    Parameters.Add( TPair<String,TValue>.Create('bias_regularizer',bias_regularizer));
    Parameters.Add( TPair<String,TValue>.Create('activity_regularizer',activity_regularizer));
    Parameters.Add( TPair<String,TValue>.Create('depthwise_constraint',depthwise_constraint));
    Parameters.Add( TPair<String,TValue>.Create('bias_constraint',bias_constraint));

    if input_shape <> nil then Parameters.Add( TPair<String,TValue>.Create('input_shape',TValue.FromShape(input_shape^)))
    else                       Parameters.Add( TPair<String,TValue>.Create('input_shape', TPythonObject.None ));

    PyInstance := GetKerasClassIstance('layers.DepthwiseConv2D');
    Init;

end;

{ TConv2DTranspose }

constructor TConv2DTranspose.Create(filters: Integer; kernel_size, strides: TArray<Integer>; padding: string;
                                    output_padding: TArray<Integer>; data_format: string; dilation_rate: TArray<Integer>; activation: string;
                                    use_bias: Boolean; kernel_initializer, bias_initializer, kernel_regularizer, bias_regularizer, activity_regularizer,
                                    kernel_constraint, bias_constraint: string; input_shape: PTnp_Shape);
var
  ksize, stride,drate : TValue;
begin
    inherited Create;

    ksize := TValue.FromShape( Tnp_Shape.Create([kernel_size[0], kernel_size[1]]) );

    if strides = nil then stride := TValue.FromShape( Tnp_Shape.Create([1, 1]) )
    else                  stride := TValue.FromShape( Tnp_Shape.Create([strides[0], strides[1]]) );

    if dilation_rate = nil then drate := TValue.FromShape( Tnp_Shape.Create([1, 1]) )
    else                        drate := TValue.FromShape( Tnp_Shape.Create([dilation_rate[0], dilation_rate[1]]) );

    Parameters.Add( TPair<String,TValue>.Create('filters',filters));
    Parameters.Add( TPair<String,TValue>.Create('kernel_size',ksize));
    Parameters.Add( TPair<String,TValue>.Create('strides',stride));
    Parameters.Add( TPair<String,TValue>.Create('padding',padding));
    Parameters.Add( TPair<String,TValue>.Create('output_padding',TValue.FromArray<Integer>(output_padding) ));
    Parameters.Add( TPair<String,TValue>.Create('data_format',data_format));
    Parameters.Add( TPair<String,TValue>.Create('dilation_rate',drate)) ;
    Parameters.Add( TPair<String,TValue>.Create('activation',activation));
    Parameters.Add( TPair<String,TValue>.Create('use_bias',use_bias));
    Parameters.Add( TPair<String,TValue>.Create('kernel_initializer',kernel_initializer));
    Parameters.Add( TPair<String,TValue>.Create('bias_initializer',bias_initializer));
    Parameters.Add( TPair<String,TValue>.Create('kernel_regularizer',kernel_regularizer));
    Parameters.Add( TPair<String,TValue>.Create('bias_regularizer',bias_regularizer));
    Parameters.Add( TPair<String,TValue>.Create('activity_regularizer',activity_regularizer));
    Parameters.Add( TPair<String,TValue>.Create('kernel_constraint',kernel_constraint));
    Parameters.Add( TPair<String,TValue>.Create('bias_constraint',bias_constraint));

    if input_shape <> nil then Parameters.Add( TPair<String,TValue>.Create('input_shape',TValue.FromShape(input_shape^)))
    else                       Parameters.Add( TPair<String,TValue>.Create('input_shape', TPythonObject.None ));

    PyInstance := GetKerasClassIstance('layers.Conv2DTranspose');
    Init;

end;

{ TConv3DTranspose }

constructor TConv3DTranspose.Create(filters: Integer; kernel_size, strides: TArray<Integer>; padding: string;
                                      output_padding: TArray<Integer>; data_format: string; dilation_rate: TArray<Integer>; activation: string;
                                      use_bias: Boolean; kernel_initializer, bias_initializer, kernel_regularizer, bias_regularizer, activity_regularizer,
                                      kernel_constraint, bias_constraint: string; input_shape: PTnp_Shape);
var
  ksize, stride,drate : TValue;
begin
    inherited Create;

    ksize := TValue.FromShape( Tnp_Shape.Create([kernel_size[0], kernel_size[1],kernel_size[2]]) );

    if strides = nil then stride := TValue.FromShape( Tnp_Shape.Create([1, 1, 1]) )
    else                  stride := TValue.FromShape( Tnp_Shape.Create([strides[0], strides[1], strides[2]]) );

    if dilation_rate = nil then drate := TValue.FromShape( Tnp_Shape.Create([1, 1, 1]) )
    else                        drate := TValue.FromShape( Tnp_Shape.Create([dilation_rate[0], dilation_rate[1], dilation_rate[2]]) );

    Parameters.Add( TPair<String,TValue>.Create('filters',filters));
    Parameters.Add( TPair<String,TValue>.Create('kernel_size',ksize));
    Parameters.Add( TPair<String,TValue>.Create('strides',stride));
    Parameters.Add( TPair<String,TValue>.Create('padding',padding));
    Parameters.Add( TPair<String,TValue>.Create('output_padding',TValue.FromArray<Integer>(output_padding) ));
    Parameters.Add( TPair<String,TValue>.Create('data_format',data_format));
    Parameters.Add( TPair<String,TValue>.Create('dilation_rate',drate));
    Parameters.Add( TPair<String,TValue>.Create('activation',activation));
    Parameters.Add( TPair<String,TValue>.Create('use_bias',use_bias));
    Parameters.Add( TPair<String,TValue>.Create('kernel_initializer',kernel_initializer));
    Parameters.Add( TPair<String,TValue>.Create('bias_initializer',bias_initializer));
    Parameters.Add( TPair<String,TValue>.Create('kernel_regularizer',kernel_regularizer));
    Parameters.Add( TPair<String,TValue>.Create('bias_regularizer',bias_regularizer));
    Parameters.Add( TPair<String,TValue>.Create('activity_regularizer',activity_regularizer));
    Parameters.Add( TPair<String,TValue>.Create('kernel_constraint',kernel_constraint));
    Parameters.Add( TPair<String,TValue>.Create('bias_constraint',bias_constraint));

    if input_shape <> nil then Parameters.Add( TPair<String,TValue>.Create('input_shape',TValue.FromShape(input_shape^)))
    else                       Parameters.Add( TPair<String,TValue>.Create('input_shape', TPythonObject.None ));

    PyInstance := GetKerasClassIstance('layers.Conv3DTranspose');
    Init;

end;

{ TCropping1D }

constructor TCropping1D.Create(cropping: TArray<Integer>; input_shape: PTnp_Shape);
var
  ksize : TValue;
begin
    inherited Create;

    ksize := TValue.FromShape( Tnp_Shape.Create([cropping[0], cropping[1]]) );

    Parameters.Add( TPair<String,TValue>.Create('cropping',ksize));

    if input_shape <> nil then Parameters.Add( TPair<String,TValue>.Create('input_shape',TValue.FromShape(input_shape^)))
    else                       Parameters.Add( TPair<String,TValue>.Create('input_shape', TPythonObject.None ));

    PyInstance := GetKerasClassIstance('layers.Cropping1D');
    Init;

end;

{ TCropping2D }

constructor TCropping2D.Create(cropping: TArray<Tnp_Shape>; data_format: string; input_shape: PTnp_Shape);
var
  ksize : TValue;
  crop  : TArray<Tnp_Shape>;
begin
    inherited Create;

    if cropping = nil then crop := [ Tnp_Shape.Create([1,1]), Tnp_Shape.Create([1,1]) ]
    else                   crop := [ Tnp_Shape.Create([cropping[0].Item[0],cropping[0].Item[1] ]),
                                     Tnp_Shape.Create([cropping[1].Item[0],cropping[1].Item[1]]) ];

    ksize := TValue.FromArray<Tnp_Shape>( crop );

    Parameters.Add( TPair<String,TValue>.Create('cropping',ksize));
    Parameters.Add( TPair<String,TValue>.Create('data_format',data_format));

    if input_shape <> nil then Parameters.Add( TPair<String,TValue>.Create('input_shape',TValue.FromShape(input_shape^)))
    else                       Parameters.Add( TPair<String,TValue>.Create('input_shape', TPythonObject.None ));

    PyInstance := GetKerasClassIstance('layers.Cropping2D');
    Init;

end;

{ TCropping3D }

constructor TCropping3D.Create(cropping: TArray<Tnp_Shape>; data_format: string; input_shape: PTnp_Shape);
var
  ksize : TValue;
  crop  : TArray<Tnp_Shape>;
begin
    inherited Create;

    if cropping = nil then crop := [ Tnp_Shape.Create([1,1]), Tnp_Shape.Create([1,1]), Tnp_Shape.Create([1,1]) ]
    else                   crop := [ Tnp_Shape.Create([ cropping[0].Item[0], cropping[0].Item[1] ]),
                                     Tnp_Shape.Create([ cropping[1].Item[0], cropping[1].Item[1] ]),
                                     Tnp_Shape.Create([ cropping[2].Item[0], cropping[2].Item[1] ]) ];

    ksize := TValue.FromArray<Tnp_Shape>( crop );

    Parameters.Add( TPair<String,TValue>.Create('cropping',ksize));
    Parameters.Add( TPair<String,TValue>.Create('data_format',data_format));

    if input_shape <> nil then Parameters.Add( TPair<String,TValue>.Create('input_shape',TValue.FromShape(input_shape^)))
    else                       Parameters.Add( TPair<String,TValue>.Create('input_shape', TPythonObject.None ));

    PyInstance := GetKerasClassIstance('layers.Cropping3D');
    Init;

end;

{ TUpSampling1D }

constructor TUpSampling1D.Create(size: Integer; input_shape: PTnp_Shape);
begin
    inherited Create;

    Parameters.Add( TPair<String,TValue>.Create('size',size));

    if input_shape <> nil then Parameters.Add( TPair<String,TValue>.Create('input_shape',TValue.FromShape(input_shape^)))
    else                       Parameters.Add( TPair<String,TValue>.Create('input_shape', TPythonObject.None ));

    PyInstance := GetKerasClassIstance('layers.UpSampling1D');
    Init;
end;

{ TUpSampling2D }

constructor TUpSampling2D.Create(size: TArray<Integer>; data_format, interpolation: string; input_shape: PTnp_Shape);
var
  ksize : TValue;
begin
    inherited Create;

    if size = nil then ksize := TValue.FromShape( Tnp_Shape.Create([2,2]) )
    else               ksize := TValue.FromShape( Tnp_Shape.Create([ size[0],size[1] ]) );

    Parameters.Add( TPair<String,TValue>.Create('size',ksize));
    Parameters.Add( TPair<String,TValue>.Create('data_format',data_format));
    Parameters.Add( TPair<String,TValue>.Create('interpolation',interpolation));

    if input_shape <> nil then Parameters.Add( TPair<String,TValue>.Create('input_shape',TValue.FromShape(input_shape^)))
    else                       Parameters.Add( TPair<String,TValue>.Create('input_shape', TPythonObject.None ));

    PyInstance := GetKerasClassIstance('layers.UpSampling2D');
    Init;
end;

{ TUpSampling3D }

constructor TUpSampling3D.Create(size: TArray<Integer>; data_format: string; input_shape: PTnp_Shape);
var
  ksize : TValue;
begin
    inherited Create;

    if size = nil then ksize := TValue.FromShape( Tnp_Shape.Create([2,2,2]) )
    else               ksize := TValue.FromShape( Tnp_Shape.Create([ size[0],size[1],size[2] ]) );

    Parameters.Add( TPair<String,TValue>.Create('size',ksize));
    Parameters.Add( TPair<String,TValue>.Create('data_format',data_format));

    if input_shape <> nil then Parameters.Add( TPair<String,TValue>.Create('input_shape',TValue.FromShape(input_shape^)))
    else                       Parameters.Add( TPair<String,TValue>.Create('input_shape', TPythonObject.None ));

    PyInstance := GetKerasClassIstance('layers.UpSampling3D');
    Init;

end;

{ TZeroPadding1D }

constructor TZeroPadding1D.Create(padding: Integer; input_shape: PTnp_Shape);
begin
    inherited Create;

    Parameters.Add( TPair<String,TValue>.Create('padding',padding));

    if input_shape <> nil then Parameters.Add( TPair<String,TValue>.Create('input_shape',TValue.FromShape(input_shape^)))
    else                       Parameters.Add( TPair<String,TValue>.Create('input_shape', TPythonObject.None ));

    PyInstance := GetKerasClassIstance('layers.ZeroPadding1D');
    Init;
end;

{ TZeroPadding2D }

constructor TZeroPadding2D.Create(padding: TArray<Integer>; data_format: string; input_shape: PTnp_Shape);
var
  ksize : TValue;
begin
    inherited Create;

    if padding = nil then ksize := TValue.FromShape( Tnp_Shape.Create([2,2]) )
    else                  ksize := TValue.FromShape( Tnp_Shape.Create([ padding[0],padding[1] ]) );

    Parameters.Add( TPair<String,TValue>.Create('padding',ksize));
    Parameters.Add( TPair<String,TValue>.Create('data_format',data_format));

    if input_shape <> nil then Parameters.Add( TPair<String,TValue>.Create('input_shape',TValue.FromShape(input_shape^)))
    else                       Parameters.Add( TPair<String,TValue>.Create('input_shape', TPythonObject.None ));

    PyInstance := GetKerasClassIstance('layers.ZeroPadding2D');
    Init;
end;

{ TZeroPadding3D }

constructor TZeroPadding3D.Create(padding: TArray<Integer>; data_format: string; input_shape: PTnp_Shape);
var
  ksize : TValue;
begin
    inherited Create;

    if padding = nil then ksize := TValue.FromShape( Tnp_Shape.Create([2,2,2]) )
    else                  ksize := TValue.FromShape( Tnp_Shape.Create([ padding[0],padding[1],padding[2] ]) );

    Parameters.Add( TPair<String,TValue>.Create('padding',ksize));
    Parameters.Add( TPair<String,TValue>.Create('data_format',data_format));

    if input_shape <> nil then Parameters.Add( TPair<String,TValue>.Create('input_shape',TValue.FromShape(input_shape^)))
    else                       Parameters.Add( TPair<String,TValue>.Create('input_shape', TPythonObject.None ));

    PyInstance := GetKerasClassIstance('layers.ZeroPadding3D');
    Init;

end;

{ TEmbedding }

constructor TEmbedding.Create(input_dim, output_dim: Integer; embeddings_initializer, embeddings_regularizer,
                                activity_regularizer, embeddings_constraint: string; mask_zero: Boolean; input_length: PInteger;
                                input_shape: PTnp_Shape);
begin
    inherited Create;

    Parameters.Add( TPair<String,TValue>.Create('input_dim',input_dim));
    Parameters.Add( TPair<String,TValue>.Create('output_dim',output_dim));
    Parameters.Add( TPair<String,TValue>.Create('embeddings_initializer',embeddings_initializer));
    Parameters.Add( TPair<String,TValue>.Create('embeddings_regularizer',embeddings_regularizer));
    Parameters.Add( TPair<String,TValue>.Create('activity_regularizer',activity_regularizer));
    Parameters.Add( TPair<String,TValue>.Create('embeddings_constraint',embeddings_constraint));
    Parameters.Add( TPair<String,TValue>.Create('mask_zero',mask_zero));

    if input_length <> nil then Parameters.Add( TPair<String,TValue>.Create('input_length',input_length^))
    else                        Parameters.Add( TPair<String,TValue>.Create('input_length', TPythonObject.None ));

    if input_shape <> nil then Parameters.Add( TPair<String,TValue>.Create('input_shape',TValue.FromShape(input_shape^)))
    else                       Parameters.Add( TPair<String,TValue>.Create('input_shape', TPythonObject.None ));


    PyInstance := GetKerasClassIstance('layers.Embedding');
    Init;
end;

constructor TEmbedding.Create(input_dim, output_dim: Integer; input_length: PInteger);
begin
    inherited Create;

    Parameters.Add( TPair<String,TValue>.Create('input_dim',input_dim));
    Parameters.Add( TPair<String,TValue>.Create('output_dim',output_dim));

    if input_length <> nil then Parameters.Add( TPair<String,TValue>.Create('input_length',input_length^))
    else                        Parameters.Add( TPair<String,TValue>.Create('input_length', TPythonObject.None ));

    PyInstance := GetKerasClassIstance('layers.Embedding');
    Init;
end;

{ TLocallyConnected1D }

constructor TLocallyConnected1D.Create(filters, kernel_size, strides: Integer; padding, data_format: string;
                                      dilation_rate: Integer; activation: string; use_bias: Boolean; kernel_initializer, bias_initializer,
                                      kernel_regularizer, bias_regularizer, activity_regularizer, kernel_constraint, bias_constraint: string;
                                      input_shape: PTnp_Shape);
begin
    inherited Create;

    Parameters.Add( TPair<String,TValue>.Create('filters',filters));
    Parameters.Add( TPair<String,TValue>.Create('kernel_size',kernel_size));
    Parameters.Add( TPair<String,TValue>.Create('strides',strides));
    Parameters.Add( TPair<String,TValue>.Create('padding',padding));
    Parameters.Add( TPair<String,TValue>.Create('data_format',data_format));
    Parameters.Add( TPair<String,TValue>.Create('dilation_rate',dilation_rate));
    Parameters.Add( TPair<String,TValue>.Create('activation',activation));
    Parameters.Add( TPair<String,TValue>.Create('use_bias',use_bias));
    Parameters.Add( TPair<String,TValue>.Create('kernel_initializer',kernel_initializer));
    Parameters.Add( TPair<String,TValue>.Create('bias_initializer',bias_initializer));
    Parameters.Add( TPair<String,TValue>.Create('kernel_regularizer',kernel_regularizer));
    Parameters.Add( TPair<String,TValue>.Create('bias_regularizer',bias_regularizer));
    Parameters.Add( TPair<String,TValue>.Create('activity_regularizer',activity_regularizer));
    Parameters.Add( TPair<String,TValue>.Create('kernel_constraint',kernel_constraint));
    Parameters.Add( TPair<String,TValue>.Create('bias_constraint',bias_constraint));

    if input_shape <> nil then Parameters.Add( TPair<String,TValue>.Create('input_shape',TValue.FromShape(input_shape^)))
    else                       Parameters.Add( TPair<String,TValue>.Create('input_shape', TPythonObject.None ));

    PyInstance := GetKerasClassIstance('layers.LocallyConnected1D');
    Init;
end;

{ TLocallyConnected2D }

constructor TLocallyConnected2D.Create(filters: Integer; kernel_size, strides: TArray<Integer>; padding,
                      data_format: string; dilation_rate: TArray<Integer>; activation: string; use_bias: Boolean; kernel_initializer,
                      bias_initializer, kernel_regularizer, bias_regularizer, activity_regularizer, kernel_constraint,
                      bias_constraint: string; input_shape: PTnp_Shape);
var
  ksize, stride,drate : TValue;
begin
    inherited Create;

    ksize := TValue.FromShape( Tnp_Shape.Create([ kernel_size[0], kernel_size[1] ]) );

    if strides = nil then stride := TValue.FromShape( Tnp_Shape.Create([1, 1]) )
    else                  stride := TValue.FromShape( Tnp_Shape.Create([ strides[0], strides[1] ]) );

    if dilation_rate = nil then drate := TValue.FromShape( Tnp_Shape.Create([1, 1]) )
    else                        drate := TValue.FromShape( Tnp_Shape.Create([ dilation_rate[0], dilation_rate[1] ]) );


    Parameters.Add( TPair<String,TValue>.Create('filters',filters));
    Parameters.Add( TPair<String,TValue>.Create('kernel_size',ksize));
    Parameters.Add( TPair<String,TValue>.Create('strides',stride));
    Parameters.Add( TPair<String,TValue>.Create('padding',padding));
    Parameters.Add( TPair<String,TValue>.Create('data_format',data_format));
    Parameters.Add( TPair<String,TValue>.Create('dilation_rate',drate));
    Parameters.Add( TPair<String,TValue>.Create('activation',activation));
    Parameters.Add( TPair<String,TValue>.Create('use_bias',use_bias));
    Parameters.Add( TPair<String,TValue>.Create('kernel_initializer',kernel_initializer));
    Parameters.Add( TPair<String,TValue>.Create('bias_initializer',bias_initializer));
    Parameters.Add( TPair<String,TValue>.Create('kernel_regularizer',kernel_regularizer));
    Parameters.Add( TPair<String,TValue>.Create('bias_regularizer',bias_regularizer));
    Parameters.Add( TPair<String,TValue>.Create('activity_regularizer',activity_regularizer));
    Parameters.Add( TPair<String,TValue>.Create('kernel_constraint',kernel_constraint));
    Parameters.Add( TPair<String,TValue>.Create('bias_constraint',bias_constraint));

    if input_shape <> nil then Parameters.Add( TPair<String,TValue>.Create('input_shape',TValue.FromShape(input_shape^)))
    else                       Parameters.Add( TPair<String,TValue>.Create('input_shape', TPythonObject.None ));

    PyInstance := GetKerasClassIstance('layers.LocallyConnected2D');
    Init;
end;

{ TMerge }

constructor TMerge.Create;
begin
    inherited Create;
end;

{ TAdd }

constructor TAdd.Create(inputs: TArray<TBaseLayer>);
begin
    inherited Create;

    Parameters.Add( TPair<String,TValue>.Create('inputs', TValue.FromArray<TBaseLayer>(inputs) ));

    PyInstance := GetKerasClassIstance('layers.add');
    Init;

end;

{ TConcatenate }

constructor TConcatenate.Create(inputs: TArray<TBaseLayer>);
begin
    inherited Create;

    Parameters.Add( TPair<String,TValue>.Create('inputs', TValue.FromArray<TBaseLayer>(inputs) ));

    PyInstance := GetKerasClassIstance('layers.concatenate');
    Init;
end;

{ TGaussianNoise }

constructor TGaussianNoise.Create(stddev: Double);
begin
    inherited Create;

    Parameters.Add( TPair<String,TValue>.Create('stddev', stddev ));

    PyInstance := GetKerasClassIstance('layers.GaussianNoise');
    Init;
end;

{ TGaussianDropout }

constructor TGaussianDropout.Create(rate: Double);
begin
    inherited Create;

    Parameters.Add( TPair<String,TValue>.Create('rate', rate ));

    PyInstance := GetKerasClassIstance('layers.GaussianDropout');
    Init;
end;

{ TAlphaDropout }

constructor TAlphaDropout.Create(rate: Double; noise_shape: TNDarray<Integer>; seed: PInteger);
begin
    inherited Create;

    Parameters.Add( TPair<String,TValue>.Create('rate', rate ));
    Parameters.Add( TPair<String,TValue>.Create('noise_shape', noise_shape ));

    if seed <> nil then Parameters.Add( TPair<String,TValue>.Create('seed',seed^))
    else                Parameters.Add( TPair<String,TValue>.Create('seed', TPythonObject.None ));

    PyInstance := GetKerasClassIstance('layers.AlphaDropout');
    Init;
end;

{ TBatchNormalization }

constructor TBatchNormalization.Create(axis: Integer; momentum, epsilon: Double; center, scale: Boolean;
                    beta_initializer, gamma_initializer, moving_mean_initializer, moving_variance_initializer, beta_regularizer,
                    gamma_regularizer, beta_constraint, gamma_constraint: string; input_shape: PTnp_Shape);
begin
     inherited Create;

     Parameters.Add( TPair<String,TValue>.Create('axis', axis ));
     Parameters.Add( TPair<String,TValue>.Create('momentum', momentum ));
     Parameters.Add( TPair<String,TValue>.Create('epsilon', epsilon ));
     Parameters.Add( TPair<String,TValue>.Create('center', center ));
     Parameters.Add( TPair<String,TValue>.Create('scale', scale ));
     Parameters.Add( TPair<String,TValue>.Create('beta_initializer', beta_initializer ));
     Parameters.Add( TPair<String,TValue>.Create('gamma_initializer', gamma_initializer ));
     Parameters.Add( TPair<String,TValue>.Create('moving_mean_initializer', moving_mean_initializer ));
     Parameters.Add( TPair<String,TValue>.Create('moving_variance_initializer', moving_variance_initializer ));
     Parameters.Add( TPair<String,TValue>.Create('beta_regularizer', beta_regularizer ));
     Parameters.Add( TPair<String,TValue>.Create('gamma_regularizer', gamma_regularizer ));
     Parameters.Add( TPair<String,TValue>.Create('beta_constraint', beta_constraint ));
     Parameters.Add( TPair<String,TValue>.Create('gamma_constraint', gamma_constraint));

     if input_shape <> nil then Parameters.Add( TPair<String,TValue>.Create('input_shape',TValue.FromShape(input_shape^)))
     else                       Parameters.Add( TPair<String,TValue>.Create('input_shape', TPythonObject.None ));

     PyInstance := GetKerasClassIstance('layers.BatchNormalization');
     Init(False);
end;

{ TMaxPooling1D }

constructor TMaxPooling1D.Create(pool_size: Integer; strides: PInteger; padding, data_format: string);
begin
    inherited Create;

    Parameters.Add( TPair<String,TValue>.Create('pool_size',pool_size));

    if strides <> nil then Parameters.Add( TPair<String,TValue>.Create('strides',strides^))
    else                   Parameters.Add( TPair<String,TValue>.Create('strides', TPythonObject.None ));

    Parameters.Add( TPair<String,TValue>.Create('padding',padding));
    Parameters.Add( TPair<String,TValue>.Create('data_format',data_format));

    PyInstance := GetKerasClassIstance('layers.MaxPooling1D');
    Init;
end;

{ TMaxPooling2D }

constructor TMaxPooling2D.Create(pool_size, strides: TArray<Integer>; padding, data_format: string);
var
  pool, stride : TValue;
begin
    inherited Create;

    if pool_size = nil then pool := TValue.FromShape( Tnp_Shape.Create([2, 2]) )
    else                    pool := TValue.FromShape( Tnp_Shape.Create([ pool_size[0], pool_size[1] ]) );

    if strides <> nil then stride := TValue.FromArray<Integer>(strides);

    Parameters.Add( TPair<String,TValue>.Create('pool_size',pool));

    if strides <> nil  then Parameters.Add( TPair<String,TValue>.Create('strides',stride))
    else                    Parameters.Add( TPair<String,TValue>.Create('strides', TPythonObject.None ));

    Parameters.Add( TPair<String,TValue>.Create('padding',padding));
    Parameters.Add( TPair<String,TValue>.Create('data_format',data_format));

    PyInstance := GetKerasClassIstance('layers.MaxPooling2D');
    Init;
end;

{ TMaxPooling3D }

constructor TMaxPooling3D.Create(pool_size, strides: TArray<Integer>; padding, data_format: string);
var
  pool, stride : TValue;
begin
    inherited Create;

    if pool_size = nil then pool := TValue.FromShape( Tnp_Shape.Create([2, 2,2]) )
    else                    pool := TValue.FromShape( Tnp_Shape.Create([ pool_size[0], pool_size[1], pool_size[2] ]) );

    if strides <> nil then stride := TValue.FromArray<Integer>(strides);

    Parameters.Add( TPair<String,TValue>.Create('pool_size',pool));

    if strides <> nil  then Parameters.Add( TPair<String,TValue>.Create('strides',stride))
    else                    Parameters.Add( TPair<String,TValue>.Create('strides', TPythonObject.None ));

    Parameters.Add( TPair<String,TValue>.Create('padding',padding));
    Parameters.Add( TPair<String,TValue>.Create('data_format',data_format));

    PyInstance := GetKerasClassIstance('layers.MaxPooling3D');
    Init;
end;

{ TAveragePooling1D }

constructor TAveragePooling1D.Create(pool_size: Integer; strides: PInteger; padding, data_format: string);
begin
    inherited Create;

    Parameters.Add( TPair<String,TValue>.Create('pool_size',pool_size));

    if strides <> nil then Parameters.Add( TPair<String,TValue>.Create('strides',strides^))
    else                   Parameters.Add( TPair<String,TValue>.Create('strides', TPythonObject.None ));

    Parameters.Add( TPair<String,TValue>.Create('padding',padding));
    Parameters.Add( TPair<String,TValue>.Create('data_format',data_format));

    PyInstance := GetKerasClassIstance('layers.AveragePooling1D');
    Init;
end;

{ TAveragePooling2D }

constructor TAveragePooling2D.Create(pool_size, strides: TArray<Integer>; padding, data_format: string);
var
  pool, stride : TValue;
begin
    inherited Create;

    if pool_size = nil then pool := TValue.FromShape( Tnp_Shape.Create([2, 2]) )
    else                    pool := TValue.FromShape( Tnp_Shape.Create([ pool_size[0], pool_size[1] ]) );

    if strides <> nil then stride := TValue.FromArray<Integer>(strides);

    Parameters.Add( TPair<String,TValue>.Create('pool_size',pool));

    if strides <> nil  then Parameters.Add( TPair<String,TValue>.Create('strides',stride))
    else                    Parameters.Add( TPair<String,TValue>.Create('strides', TPythonObject.None ));

    Parameters.Add( TPair<String,TValue>.Create('padding',padding));
    Parameters.Add( TPair<String,TValue>.Create('data_format',data_format));

    PyInstance := GetKerasClassIstance('layers.AveragePooling2D');
    Init;

end;

{ TAveragePooling3D }

constructor TAveragePooling3D.Create(pool_size, strides: TArray<Integer>; padding, data_format: string);
var
  pool, stride : TValue;
begin
    inherited Create;

    if pool_size = nil then pool := TValue.FromShape( Tnp_Shape.Create([2, 2, 2]) )
    else                    pool := TValue.FromShape( Tnp_Shape.Create([ pool_size[0], pool_size[1], pool_size[2] ]) );

    if strides <> nil then stride := TValue.FromArray<Integer>(strides);

    Parameters.Add( TPair<String,TValue>.Create('pool_size',pool));

    if strides <> nil  then Parameters.Add( TPair<String,TValue>.Create('strides',stride))
    else                    Parameters.Add( TPair<String,TValue>.Create('strides', TPythonObject.None ));

    Parameters.Add( TPair<String,TValue>.Create('padding',padding));
    Parameters.Add( TPair<String,TValue>.Create('data_format',data_format));

    PyInstance := GetKerasClassIstance('layers.AveragePooling3D');
    Init;
end;

{ TGlobalMaxPooling1D }

constructor TGlobalMaxPooling1D.Create(data_format: string);
begin
    inherited Create;

    Parameters.Add( TPair<String,TValue>.Create('data_format',data_format));

    PyInstance := GetKerasClassIstance('layers.GlobalMaxPooling1D');
    Init;
end;

{ TGlobalMaxPooling2D }

constructor TGlobalMaxPooling2D.Create(data_format: string);
begin
    inherited Create;

    Parameters.Add( TPair<String,TValue>.Create('data_format',data_format));

    PyInstance := GetKerasClassIstance('layers.GlobalMaxPooling2D');
    Init;
end;

{ TGlobalMaxPooling3D }

constructor TGlobalMaxPooling3D.Create(data_format: string);
begin
    inherited Create;

    Parameters.Add( TPair<String,TValue>.Create('data_format',data_format));

    PyInstance := GetKerasClassIstance('layers.GlobalMaxPooling3D');
    Init;
end;

{ TGlobalAveragePooling1D }

constructor TGlobalAveragePooling1D.Create(data_format: string);
begin
    inherited Create;

    Parameters.Add( TPair<String,TValue>.Create('data_format',data_format));

    PyInstance := GetKerasClassIstance('layers.GlobalAveragePooling1D');
    Init;
end;

{ TGlobalAveragePooling2D }

constructor TGlobalAveragePooling2D.Create(data_format: string);
begin
    inherited Create;

    Parameters.Add( TPair<String,TValue>.Create('data_format',data_format));

    PyInstance := GetKerasClassIstance('layers.GlobalAveragePooling2D');
    Init;
end;

{ TGlobalAveragePooling3D }

constructor TGlobalAveragePooling3D.Create(data_format: string);
begin
    inherited Create;

    Parameters.Add( TPair<String,TValue>.Create('data_format',data_format));

    PyInstance := GetKerasClassIstance('layers.GlobalAveragePooling3D');
    Init;
end;

{ TTimeDistributed }

constructor TTimeDistributed.Create(layer: TBaseLayer);
begin
    inherited Create;

    Parameters.Add( TPair<String,TValue>.Create('layer',layer.PyInstance));

    PyInstance := GetKerasClassIstance('layers.TimeDistributed');
    Init;
end;

{ TBidirectional }

constructor TBidirectional.Create(layer: TBaseLayer; merge_mode: string; weights: TNDArray; input_shape: PTnp_Shape);
begin
    inherited Create;

    Parameters.Add( TPair<String,TValue>.Create('layer',layer.PyInstance));
    Parameters.Add( TPair<String,TValue>.Create('merge_mode',merge_mode));
    Parameters.Add( TPair<String,TValue>.Create('weights',weights));

    if input_shape <> nil then Parameters.Add( TPair<String,TValue>.Create('input_shape',TValue.FromShape(input_shape^)))
    else                       Parameters.Add( TPair<String,TValue>.Create('input_shape', TPythonObject.None ));

    PyInstance := GetKerasClassIstance('layers.Bidirectional');
    Init;
end;

constructor TBidirectional.Create(layer: TBaseLayer; input_shape: PTnp_Shape);
begin
    inherited Create;

    Parameters.Add( TPair<String,TValue>.Create('layer',layer.PyInstance));

    if input_shape <> nil then Parameters.Add( TPair<String,TValue>.Create('input_shape',TValue.FromShape(input_shape^)))
    else                       Parameters.Add( TPair<String,TValue>.Create('input_shape', TPythonObject.None ));

    PyInstance := GetKerasClassIstance('layers.Bidirectional');
    Init;
end;

{ TRNN }

constructor TRNN.Create(cell: TRNN; return_sequences, return_state, go_backwards, stateful, unroll: Boolean; input_dim,
                               input_length: PInteger; input_shape: PTnp_Shape);
begin
    Parameters := TList< TPair<String,TValue> >.Create;

    Parameters.Add( TPair<String,TValue>.Create('cell',cell.PyInstance));
    Parameters.Add( TPair<String,TValue>.Create('return_sequences',return_sequences));
    Parameters.Add( TPair<String,TValue>.Create('return_state',return_state));
    Parameters.Add( TPair<String,TValue>.Create('go_backwards',go_backwards));
    Parameters.Add( TPair<String,TValue>.Create('stateful',stateful));
    Parameters.Add( TPair<String,TValue>.Create('unroll',unroll));

    if input_length <> nil then Parameters.Add( TPair<String,TValue>.Create('input_length',input_length^))
    else                        Parameters.Add( TPair<String,TValue>.Create('input_length', TPythonObject.None ));

    if input_shape <> nil then Parameters.Add( TPair<String,TValue>.Create('input_shape',TValue.FromShape(input_shape^)))
    else                       Parameters.Add( TPair<String,TValue>.Create('input_shape', TPythonObject.None ));

    PyInstance := GetKerasClassIstance('layers.RNN');
    Init;
end;

{ TSimpleRNN }

constructor TSimpleRNN.Create(units: Integer; activation: string; use_bias: Boolean; kernel_initializer,
                                recurrent_initializer, bias_initializer, kernel_regularizer, recurrent_regularizer, bias_regularizer,
                                activity_regularizer, kernel_constraint, recurrent_constraint, bias_constraint: string; dropout,
                                recurrent_dropout: Double; return_sequences, return_state, go_backwards, stateful, unroll: Boolean);
begin
    Parameters := TList< TPair<String,TValue> >.Create;

    Parameters.Add( TPair<String,TValue>.Create('units',units));
    Parameters.Add( TPair<String,TValue>.Create('activation',activation));
    Parameters.Add( TPair<String,TValue>.Create('use_bias',use_bias));
    Parameters.Add( TPair<String,TValue>.Create('kernel_initializer',kernel_initializer));
    Parameters.Add( TPair<String,TValue>.Create('recurrent_initializer',recurrent_initializer));
    Parameters.Add( TPair<String,TValue>.Create('bias_initializer',bias_initializer));
    Parameters.Add( TPair<String,TValue>.Create('kernel_regularizer',kernel_regularizer));
    Parameters.Add( TPair<String,TValue>.Create('recurrent_regularizer',recurrent_regularizer));
    Parameters.Add( TPair<String,TValue>.Create('bias_regularizer',bias_regularizer));
    Parameters.Add( TPair<String,TValue>.Create('activity_regularizer',activity_regularizer));
    Parameters.Add( TPair<String,TValue>.Create('kernel_constraint',kernel_constraint));
    Parameters.Add( TPair<String,TValue>.Create('recurrent_constraint',recurrent_constraint));
    Parameters.Add( TPair<String,TValue>.Create('bias_constraint',bias_constraint));
    Parameters.Add( TPair<String,TValue>.Create('dropout',dropout));
    Parameters.Add( TPair<String,TValue>.Create('recurrent_dropout',recurrent_dropout));
    Parameters.Add( TPair<String,TValue>.Create('return_sequences',return_sequences));
    Parameters.Add( TPair<String,TValue>.Create('return_state',return_state));
    Parameters.Add( TPair<String,TValue>.Create('go_backwards',go_backwards));
    Parameters.Add( TPair<String,TValue>.Create('stateful',stateful));
    Parameters.Add( TPair<String,TValue>.Create('unroll',unroll));

    PyInstance := GetKerasClassIstance('layers.SimpleRNN');
    Init;
end;

{ TSimpleRNNCell }

constructor TSimpleRNNCell.Create(units: Integer; activation: string; use_bias: Boolean; kernel_initializer,
                                recurrent_initializer, bias_initializer, kernel_regularizer, recurrent_regularizer, bias_regularizer,
                                activity_regularizer, kernel_constraint, recurrent_constraint, bias_constraint: string; dropout,
                                recurrent_dropout: Double);
begin
    Parameters := TList< TPair<String,TValue> >.Create;

    Parameters.Add( TPair<String,TValue>.Create('units',units));
    Parameters.Add( TPair<String,TValue>.Create('activation',activation));
    Parameters.Add( TPair<String,TValue>.Create('use_bias',use_bias));
    Parameters.Add( TPair<String,TValue>.Create('kernel_initializer',kernel_initializer));
    Parameters.Add( TPair<String,TValue>.Create('recurrent_initializer',recurrent_initializer));
    Parameters.Add( TPair<String,TValue>.Create('bias_initializer',bias_initializer));
    Parameters.Add( TPair<String,TValue>.Create('kernel_regularizer',kernel_regularizer));
    Parameters.Add( TPair<String,TValue>.Create('recurrent_regularizer',recurrent_regularizer));
    Parameters.Add( TPair<String,TValue>.Create('bias_regularizer',bias_regularizer));
    Parameters.Add( TPair<String,TValue>.Create('activity_regularizer',activity_regularizer));
    Parameters.Add( TPair<String,TValue>.Create('kernel_constraint',kernel_constraint));
    Parameters.Add( TPair<String,TValue>.Create('recurrent_constraint',recurrent_constraint));
    Parameters.Add( TPair<String,TValue>.Create('bias_constraint',bias_constraint));
    Parameters.Add( TPair<String,TValue>.Create('dropout',dropout));
    Parameters.Add( TPair<String,TValue>.Create('recurrent_dropout',recurrent_dropout));

    PyInstance := GetKerasClassIstance('layers.SimpleRNNCell');
    Init;
end;

{ TGRU }

constructor TGRU.Create(units: Integer; activation,recurrent_activation: string; use_bias: Boolean; kernel_initializer,
                    recurrent_initializer, bias_initializer, kernel_regularizer, recurrent_regularizer, bias_regularizer,
                    activity_regularizer, kernel_constraint, recurrent_constraint, bias_constraint: string; dropout,
                    recurrent_dropout: Double; implement: Integer; return_sequences, return_state, go_backwards, stateful, unroll,
                    reset_after: Boolean);
begin
    Parameters := TList< TPair<String,TValue> >.Create;

    Parameters.Add( TPair<String,TValue>.Create('units',units));
    Parameters.Add( TPair<String,TValue>.Create('activation',activation));
    Parameters.Add( TPair<String,TValue>.Create('recurrent_activation',recurrent_activation));
    Parameters.Add( TPair<String,TValue>.Create('use_bias',use_bias));
    Parameters.Add( TPair<String,TValue>.Create('kernel_initializer',kernel_initializer));
    Parameters.Add( TPair<String,TValue>.Create('recurrent_initializer',recurrent_initializer));
    Parameters.Add( TPair<String,TValue>.Create('bias_initializer',bias_initializer));
    Parameters.Add( TPair<String,TValue>.Create('kernel_regularizer',kernel_regularizer));
    Parameters.Add( TPair<String,TValue>.Create('recurrent_regularizer',recurrent_regularizer));
    Parameters.Add( TPair<String,TValue>.Create('bias_regularizer',bias_regularizer));
    Parameters.Add( TPair<String,TValue>.Create('activity_regularizer',activity_regularizer));
    Parameters.Add( TPair<String,TValue>.Create('kernel_constraint',kernel_constraint));
    Parameters.Add( TPair<String,TValue>.Create('recurrent_constraint',recurrent_constraint));
    Parameters.Add( TPair<String,TValue>.Create('bias_constraint',bias_constraint));
    Parameters.Add( TPair<String,TValue>.Create('dropout',dropout));
    Parameters.Add( TPair<String,TValue>.Create('recurrent_dropout',recurrent_dropout));
    Parameters.Add( TPair<String,TValue>.Create('implementation',implement));
    Parameters.Add( TPair<String,TValue>.Create('return_sequences',return_sequences));
    Parameters.Add( TPair<String,TValue>.Create('return_state',return_state));
    Parameters.Add( TPair<String,TValue>.Create('go_backwards',go_backwards));
    Parameters.Add( TPair<String,TValue>.Create('stateful',stateful));
    Parameters.Add( TPair<String,TValue>.Create('unroll',unroll));
    Parameters.Add( TPair<String,TValue>.Create('reset_after',reset_after));

    PyInstance := GetKerasClassIstance('layers.GRU');
    Init;
end;

constructor TGRU.Create(units: Integer; dropout, recurrent_dropout: Double);
begin
    Parameters := TList< TPair<String,TValue> >.Create;

    Parameters.Add( TPair<String,TValue>.Create('units',units));
    Parameters.Add( TPair<String,TValue>.Create('dropout',dropout));
    Parameters.Add( TPair<String,TValue>.Create('recurrent_dropout',recurrent_dropout));

    PyInstance := GetKerasClassIstance('layers.GRU');
    Init;
end;

{ TCuDNNGRU }

constructor TCuDNNGRU.Create(units: Integer; kernel_initializer, recurrent_initializer, bias_initializer,
                          kernel_regularizer, recurrent_regularizer, bias_regularizer, activity_regularizer, kernel_constraint,
                          recurrent_constraint, bias_constraint: string; return_sequences, return_state, stateful: Boolean);
begin
    Parameters := TList< TPair<String,TValue> >.Create;

    Parameters.Add( TPair<String,TValue>.Create('units',units));
    Parameters.Add( TPair<String,TValue>.Create('kernel_initializer',kernel_initializer));
    Parameters.Add( TPair<String,TValue>.Create('recurrent_initializer',recurrent_initializer));
    Parameters.Add( TPair<String,TValue>.Create('bias_initializer',bias_initializer));
    Parameters.Add( TPair<String,TValue>.Create('kernel_regularizer',kernel_regularizer));
    Parameters.Add( TPair<String,TValue>.Create('recurrent_regularizer',recurrent_regularizer));
    Parameters.Add( TPair<String,TValue>.Create('bias_regularizer',bias_regularizer));
    Parameters.Add( TPair<String,TValue>.Create('activity_regularizer',activity_regularizer));
    Parameters.Add( TPair<String,TValue>.Create('kernel_constraint',kernel_constraint));
    Parameters.Add( TPair<String,TValue>.Create('recurrent_constraint',recurrent_constraint));
    Parameters.Add( TPair<String,TValue>.Create('bias_constraint',bias_constraint));
    Parameters.Add( TPair<String,TValue>.Create('return_sequences',return_sequences));
    Parameters.Add( TPair<String,TValue>.Create('return_state',return_state));
    Parameters.Add( TPair<String,TValue>.Create('stateful',stateful));

    PyInstance := GetKerasClassIstance('layers.CuDNNGRU');
    Init;

end;

{ TGRUCell }

constructor TGRUCell.Create(units: Integer; activation, recurrent_activation: string; use_bias: Boolean;
                      kernel_initializer, recurrent_initializer, bias_initializer, kernel_regularizer, recurrent_regularizer,
                      bias_regularizer, kernel_constraint, recurrent_constraint, bias_constraint: string; dropout,
                      recurrent_dropout: Double; implement: Integer; reset_after: Boolean);
begin
    Parameters := TList< TPair<String,TValue> >.Create;

    Parameters.Add( TPair<String,TValue>.Create('units',units));
    Parameters.Add( TPair<String,TValue>.Create('activation',activation));
    Parameters.Add( TPair<String,TValue>.Create('recurrent_activation',recurrent_activation));
    Parameters.Add( TPair<String,TValue>.Create('use_bias',use_bias));
    Parameters.Add( TPair<String,TValue>.Create('kernel_initializer',kernel_initializer));
    Parameters.Add( TPair<String,TValue>.Create('recurrent_initializer',recurrent_initializer));
    Parameters.Add( TPair<String,TValue>.Create('bias_initializer',bias_initializer));
    Parameters.Add( TPair<String,TValue>.Create('kernel_regularizer',kernel_regularizer));
    Parameters.Add( TPair<String,TValue>.Create('recurrent_regularizer',recurrent_regularizer));
    Parameters.Add( TPair<String,TValue>.Create('bias_regularizer',bias_regularizer));
    Parameters.Add( TPair<String,TValue>.Create('kernel_constraint',kernel_constraint));
    Parameters.Add( TPair<String,TValue>.Create('recurrent_constraint',recurrent_constraint));
    Parameters.Add( TPair<String,TValue>.Create('bias_constraint',bias_constraint));
    Parameters.Add( TPair<String,TValue>.Create('dropout',dropout));
    Parameters.Add( TPair<String,TValue>.Create('recurrent_dropout',recurrent_dropout));
    Parameters.Add( TPair<String,TValue>.Create('implementation',implement));
    Parameters.Add( TPair<String,TValue>.Create('reset_after',reset_after));

    PyInstance := GetKerasClassIstance('layers.GRUCell');
    Init;
end;

{ TLSTM }

constructor TLSTM.Create(units: Integer; activation, recurrent_activation: string; use_bias: Boolean;
                            kernel_initializer, recurrent_initializer, bias_initializer: string; unit_forget_bias: Boolean; kernel_regularizer,
                            recurrent_regularizer, bias_regularizer, activity_regularizer, kernel_constraint, recurrent_constraint,
                            bias_constraint: string; dropout, recurrent_dropout: Double; implement: Integer; return_sequences, return_state,
                            go_backwards, stateful, unroll: Boolean; input_shape: PTnp_Shape);
begin
    Parameters := TList< TPair<String,TValue> >.Create;

    Parameters.Add( TPair<String,TValue>.Create('units',units));
    Parameters.Add( TPair<String,TValue>.Create('activation',activation));
    Parameters.Add( TPair<String,TValue>.Create('recurrent_activation',recurrent_activation));
    Parameters.Add( TPair<String,TValue>.Create('use_bias',use_bias));
    Parameters.Add( TPair<String,TValue>.Create('kernel_initializer',kernel_initializer));
    Parameters.Add( TPair<String,TValue>.Create('recurrent_initializer',recurrent_initializer));
    Parameters.Add( TPair<String,TValue>.Create('bias_initializer',bias_initializer));
    Parameters.Add( TPair<String,TValue>.Create('unit_forget_bias',unit_forget_bias));
    Parameters.Add( TPair<String,TValue>.Create('kernel_regularizer',kernel_regularizer));
    Parameters.Add( TPair<String,TValue>.Create('recurrent_regularizer',recurrent_regularizer));
    Parameters.Add( TPair<String,TValue>.Create('bias_regularizer',bias_regularizer));
    Parameters.Add( TPair<String,TValue>.Create('activity_regularizer',activity_regularizer));
    Parameters.Add( TPair<String,TValue>.Create('kernel_constraint',kernel_constraint));
    Parameters.Add( TPair<String,TValue>.Create('recurrent_constraint',recurrent_constraint));
    Parameters.Add( TPair<String,TValue>.Create('bias_constraint',bias_constraint));
    Parameters.Add( TPair<String,TValue>.Create('dropout',dropout));
    Parameters.Add( TPair<String,TValue>.Create('recurrent_dropout',recurrent_dropout));
    Parameters.Add( TPair<String,TValue>.Create('implementation',implement));
    Parameters.Add( TPair<String,TValue>.Create('return_sequences',return_sequences));
    Parameters.Add( TPair<String,TValue>.Create('return_state',return_state));
    Parameters.Add( TPair<String,TValue>.Create('go_backwards',go_backwards));
    Parameters.Add( TPair<String,TValue>.Create('stateful',stateful));
    Parameters.Add( TPair<String,TValue>.Create('unroll',unroll));

     if input_shape <> nil then Parameters.Add( TPair<String,TValue>.Create('input_shape',TValue.FromShape(input_shape^)))
    else                       Parameters.Add( TPair<String,TValue>.Create('input_shape', TPythonObject.None ));

    PyInstance := GetKerasClassIstance('layers.LSTM');
    Init;
end;

constructor TLSTM.Create(units: Integer; dropout, recurrent_dropout: Double);
begin
    Parameters := TList< TPair<String,TValue> >.Create;

    Parameters.Add( TPair<String,TValue>.Create('units',units));
    Parameters.Add( TPair<String,TValue>.Create('dropout',dropout));
    Parameters.Add( TPair<String,TValue>.Create('recurrent_dropout',recurrent_dropout));

    PyInstance := GetKerasClassIstance('layers.LSTM');
    Init;
end;

constructor TLSTM.Create(units: Integer; activation: string; input_shape : PTnp_Shape; stateful: Boolean; return_sequences: Boolean);
begin
    Parameters := TList< TPair<String,TValue> >.Create;

    Parameters.Add( TPair<String,TValue>.Create('units',units));
    Parameters.Add( TPair<String,TValue>.Create('activation',activation));

    if input_shape <> nil then Parameters.Add( TPair<String,TValue>.Create('input_shape',TValue.FromShape(input_shape^)))
    else                       Parameters.Add( TPair<String,TValue>.Create('input_shape', TPythonObject.None ));

    Parameters.Add( TPair<String,TValue>.Create('stateful',stateful));
    Parameters.Add( TPair<String,TValue>.Create('return_sequences',return_sequences));

    PyInstance := GetKerasClassIstance('layers.LSTM');
    Init;
end;

{ TCuDNNLSTM }

constructor TCuDNNLSTM.Create(units: Integer; kernel_initializer, recurrent_initializer, bias_initializer: string;
                      unit_forget_bias: Boolean; kernel_regularizer, recurrent_regularizer, bias_regularizer, activity_regularizer,
                      kernel_constraint, recurrent_constraint, bias_constraint: string; return_sequences, return_state, stateful: Boolean);
begin
    Parameters := TList< TPair<String,TValue> >.Create;

    Parameters.Add( TPair<String,TValue>.Create('units',units));
    Parameters.Add( TPair<String,TValue>.Create('kernel_initializer',kernel_initializer));
    Parameters.Add( TPair<String,TValue>.Create('recurrent_initializer',recurrent_initializer));
    Parameters.Add( TPair<String,TValue>.Create('bias_initializer',bias_initializer));
    Parameters.Add( TPair<String,TValue>.Create('unit_forget_bias',unit_forget_bias));
    Parameters.Add( TPair<String,TValue>.Create('kernel_regularizer',kernel_regularizer));
    Parameters.Add( TPair<String,TValue>.Create('recurrent_regularizer',recurrent_regularizer));
    Parameters.Add( TPair<String,TValue>.Create('bias_regularizer',bias_regularizer));
    Parameters.Add( TPair<String,TValue>.Create('activity_regularizer',activity_regularizer));
    Parameters.Add( TPair<String,TValue>.Create('kernel_constraint',kernel_constraint));
    Parameters.Add( TPair<String,TValue>.Create('recurrent_constraint',recurrent_constraint));
    Parameters.Add( TPair<String,TValue>.Create('bias_constraint',bias_constraint));
    Parameters.Add( TPair<String,TValue>.Create('return_sequences',return_sequences));
    Parameters.Add( TPair<String,TValue>.Create('return_state',return_state));
    Parameters.Add( TPair<String,TValue>.Create('stateful',stateful));

    // not work for tf ver 2
    PyInstance := GetKerasClassIstance('layers.CuDNNLSTM');
    PyInstance := GetTFClassIstance('compat.v1.keras.layers.CuDNNLSTM');
    Init;


end;

{ TLSTMCell }

constructor TLSTMCell.Create(units: Integer; activation, recurrent_activation: string; use_bias: Boolean;
                              kernel_initializer, recurrent_initializer, bias_initializer: string; unit_forget_bias: Boolean; kernel_regularizer,
                              recurrent_regularizer, bias_regularizer, kernel_constraint, recurrent_constraint, bias_constraint: string; dropout,
                              recurrent_dropout: Double; implement: Integer);
begin
    Parameters := TList< TPair<String,TValue> >.Create;

    Parameters.Add( TPair<String,TValue>.Create('units',units));
    Parameters.Add( TPair<String,TValue>.Create('activation',activation));
    Parameters.Add( TPair<String,TValue>.Create('recurrent_activation',recurrent_activation));
    Parameters.Add( TPair<String,TValue>.Create('use_bias',use_bias));
    Parameters.Add( TPair<String,TValue>.Create('kernel_initializer',kernel_initializer));
    Parameters.Add( TPair<String,TValue>.Create('recurrent_initializer',recurrent_initializer));
    Parameters.Add( TPair<String,TValue>.Create('bias_initializer',bias_initializer));
    Parameters.Add( TPair<String,TValue>.Create('unit_forget_bias',unit_forget_bias));
    Parameters.Add( TPair<String,TValue>.Create('kernel_regularizer',kernel_regularizer));
    Parameters.Add( TPair<String,TValue>.Create('recurrent_regularizer',recurrent_regularizer));
    Parameters.Add( TPair<String,TValue>.Create('bias_regularizer',bias_regularizer));
    Parameters.Add( TPair<String,TValue>.Create('kernel_constraint',kernel_constraint));
    Parameters.Add( TPair<String,TValue>.Create('recurrent_constraint',recurrent_constraint));
    Parameters.Add( TPair<String,TValue>.Create('bias_constraint',bias_constraint));
    Parameters.Add( TPair<String,TValue>.Create('dropout',dropout));
    Parameters.Add( TPair<String,TValue>.Create('recurrent_dropout',recurrent_dropout));
    Parameters.Add( TPair<String,TValue>.Create('implementation',implement));

    PyInstance := GetKerasClassIstance('layers.LSTMCell');
    Init;
end;

{ TConvLSTM2D }

constructor TConvLSTM2D.Create(filters: Integer; kernel_size, strides: TArray<Integer>; padding, data_format: string;
                                dilation_rate: TArray<Integer>; activation, recurrent_activation: string; use_bias: Boolean; kernel_initializer,
                                recurrent_initializer, bias_initializer: string; unit_forget_bias: Boolean; kernel_regularizer, recurrent_regularizer,
                                bias_regularizer, activity_regularizer, kernel_constraint, recurrent_constraint, bias_constraint: string;
                                return_sequences, go_backwards, stateful: Boolean; dropout, recurrent_dropout: Double; input_shape: PTnp_Shape);
var
  ksize, stride,drate : TValue;
begin
    Parameters := TList< TPair<String,TValue> >.Create;

    ksize := TValue.FromShape( Tnp_Shape.Create([ kernel_size[0], kernel_size[1] ]) );

    if strides = nil then stride := TValue.FromShape( Tnp_Shape.Create([1, 1]) )
    else                  stride := TValue.FromShape( Tnp_Shape.Create([ strides[0], strides[1] ]) );

    if dilation_rate = nil then drate := TValue.FromShape( Tnp_Shape.Create([1, 1]) )
    else                        drate := TValue.FromShape( Tnp_Shape.Create([ dilation_rate[0], dilation_rate[1] ]) );

    Parameters.Add( TPair<String,TValue>.Create('filters',filters));
    Parameters.Add( TPair<String,TValue>.Create('kernel_size',ksize));
    Parameters.Add( TPair<String,TValue>.Create('strides',stride));
    Parameters.Add( TPair<String,TValue>.Create('padding',padding));
    Parameters.Add( TPair<String,TValue>.Create('data_format',data_format));
    Parameters.Add( TPair<String,TValue>.Create('dilation_rate',drate));
    Parameters.Add( TPair<String,TValue>.Create('activation',activation));
    Parameters.Add( TPair<String,TValue>.Create('recurrent_activation',recurrent_activation));
    Parameters.Add( TPair<String,TValue>.Create('use_bias',use_bias));
    Parameters.Add( TPair<String,TValue>.Create('kernel_initializer',kernel_initializer));
    Parameters.Add( TPair<String,TValue>.Create('recurrent_initializer',recurrent_initializer));
    Parameters.Add( TPair<String,TValue>.Create('bias_initializer',bias_initializer));
    Parameters.Add( TPair<String,TValue>.Create('unit_forget_bias',unit_forget_bias));
    Parameters.Add( TPair<String,TValue>.Create('kernel_regularizer',kernel_regularizer));
    Parameters.Add( TPair<String,TValue>.Create('recurrent_regularizer',recurrent_regularizer));
    Parameters.Add( TPair<String,TValue>.Create('bias_regularizer',bias_regularizer));
    Parameters.Add( TPair<String,TValue>.Create('activity_regularizer',activity_regularizer));
    Parameters.Add( TPair<String,TValue>.Create('kernel_constraint',kernel_constraint));
    Parameters.Add( TPair<String,TValue>.Create('recurrent_constraint',recurrent_constraint));
    Parameters.Add( TPair<String,TValue>.Create('bias_constraint',bias_constraint));
    Parameters.Add( TPair<String,TValue>.Create('return_sequences',return_sequences));
    Parameters.Add( TPair<String,TValue>.Create('go_backwards',go_backwards));
    Parameters.Add( TPair<String,TValue>.Create('stateful',stateful));
    Parameters.Add( TPair<String,TValue>.Create('dropout',dropout));
    Parameters.Add( TPair<String,TValue>.Create('recurrent_dropout',recurrent_dropout));

    if input_shape <> nil then Parameters.Add( TPair<String,TValue>.Create('input_shape',TValue.FromShape(input_shape^)))
    else                       Parameters.Add( TPair<String,TValue>.Create('input_shape', TPythonObject.None ));

    PyInstance := GetKerasClassIstance('layers.ConvLSTM2D');
    Init;
end;

{ TConvLSTM2DCell }

constructor TConvLSTM2DCell.Create(filters: Integer; kernel_size, strides: TArray<Integer>; padding,
                                    data_format: string; dilation_rate: TArray<Integer>; activation, recurrent_activation: string; use_bias: Boolean;
                                    kernel_initializer, recurrent_initializer, bias_initializer: string; unit_forget_bias: Boolean; kernel_regularizer,
                                    recurrent_regularizer, bias_regularizer, kernel_constraint, recurrent_constraint, bias_constraint: string; dropout,
                                    recurrent_dropout: Double);
var
  ksize, stride,drate : TValue;
begin
    Parameters := TList< TPair<String,TValue> >.Create;

    ksize := TValue.FromShape( Tnp_Shape.Create([ kernel_size[0], kernel_size[1] ]) );

    if strides = nil then stride := TValue.FromShape( Tnp_Shape.Create([1, 1]) )
    else                  stride := TValue.FromShape( Tnp_Shape.Create([ strides[0], strides[1] ]) );

    if dilation_rate = nil then drate := TValue.FromShape( Tnp_Shape.Create([1, 1]) )
    else                        drate := TValue.FromShape( Tnp_Shape.Create([ dilation_rate[0], dilation_rate[1] ]) );

    Parameters.Add( TPair<String,TValue>.Create('filters',filters));
    Parameters.Add( TPair<String,TValue>.Create('kernel_size',ksize));
    Parameters.Add( TPair<String,TValue>.Create('strides',stride));
    Parameters.Add( TPair<String,TValue>.Create('padding',padding));
    Parameters.Add( TPair<String,TValue>.Create('data_format',data_format));
    Parameters.Add( TPair<String,TValue>.Create('dilation_rate',drate));
    Parameters.Add( TPair<String,TValue>.Create('activation',activation));
    Parameters.Add( TPair<String,TValue>.Create('recurrent_activation',recurrent_activation));
    Parameters.Add( TPair<String,TValue>.Create('use_bias',use_bias));
    Parameters.Add( TPair<String,TValue>.Create('kernel_initializer',kernel_initializer));
    Parameters.Add( TPair<String,TValue>.Create('recurrent_initializer',recurrent_initializer));
    Parameters.Add( TPair<String,TValue>.Create('bias_initializer',bias_initializer));
    Parameters.Add( TPair<String,TValue>.Create('unit_forget_bias',unit_forget_bias));
    Parameters.Add( TPair<String,TValue>.Create('kernel_regularizer',kernel_regularizer));
    Parameters.Add( TPair<String,TValue>.Create('recurrent_regularizer',recurrent_regularizer));
    Parameters.Add( TPair<String,TValue>.Create('bias_regularizer',bias_regularizer));
    Parameters.Add( TPair<String,TValue>.Create('kernel_constraint',kernel_constraint));
    Parameters.Add( TPair<String,TValue>.Create('recurrent_constraint',recurrent_constraint));
    Parameters.Add( TPair<String,TValue>.Create('bias_constraint',bias_constraint));
    Parameters.Add( TPair<String,TValue>.Create('dropout',dropout));
    Parameters.Add( TPair<String,TValue>.Create('recurrent_dropout',recurrent_dropout));

    PyInstance := GetKerasClassIstance('layers.ConvLSTM2DCell');
    Init;
end;

end.
