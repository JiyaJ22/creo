import React, { useState, useRef, useEffect } from "react";
import { Camera, Upload, Zap, AlertCircle, CheckCircle, Info, Brain, ChevronDown, ChevronUp } from "lucide-react";
import * as tf from '@tensorflow/tfjs';
import axios from "axios";

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const ImageClassifier = () => {
  const [selectedImage, setSelectedImage] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [model, setModel] = useState(null);
  const [modelLoading, setModelLoading] = useState(false);
  const [dataInputs, setDataInputs] = useState({
    sqft: "",
    bed: "",
    bath: "",
    city: ""
  });
  const [combinedPrediction, setCombinedPrediction] = useState(null);
  const [isAIExplanationOpen, setIsAIExplanationOpen] = useState(false);
  const fileInputRef = useRef(null);

  // Load Teachable Machine model
  useEffect(() => {
    loadModel();
  }, []);

  const loadModel = async () => {
    try {
      setModelLoading(true);
      const modelURL = 'https://teachablemachine.withgoogle.com/models/KjXP4uvx0/';
      const loadedModel = await tf.loadLayersModel(modelURL + 'model.json');
      setModel(loadedModel);
    } catch (error) {
      console.error('Error loading model:', error);
      setError('Failed to load ML model. Image classification may not work properly.');
    } finally {
      setModelLoading(false);
    }
  };

  const handleImageUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      setSelectedImage(file);
      setPrediction(null);
      setCombinedPrediction(null);
      setError(null);
      
      // Create image preview
      const reader = new FileReader();
      reader.onload = (e) => {
        setImagePreview(e.target.result);
      };
      reader.readAsDataURL(file);
    }
  };

  const preprocessImage = (imageElement) => {
    return tf.tidy(() => {
      // Convert image to tensor
      const imageTensor = tf.browser.fromPixels(imageElement);
      
      // Resize to 224x224 (typical for Teachable Machine models)
      const resized = tf.image.resizeBilinear(imageTensor, [224, 224]);
      
      // Normalize pixel values to [0, 1]
      const normalized = resized.div(255.0);
      
      // Add batch dimension
      const batched = normalized.expandDims(0);
      
      return batched;
    });
  };

  const classifyImage = async () => {
    if (!selectedImage || !model) return;
    
    try {
      setLoading(true);
      setError(null);
      
      // Create image element for preprocessing
      const imageElement = new Image();
      imageElement.crossOrigin = 'anonymous';
      
      await new Promise((resolve, reject) => {
        imageElement.onload = resolve;
        imageElement.onerror = reject;
        imageElement.src = imagePreview;
      });
      
      // Preprocess image
      const preprocessedImage = preprocessImage(imageElement);
      
      // Make prediction
      const predictions = await model.predict(preprocessedImage).data();
      
      // Clean up tensors
      preprocessedImage.dispose();
      
      // Interpret predictions (adjust based on your model's output)
      const classes = ['Low', 'Mid', 'High'];
      const maxIndex = predictions.indexOf(Math.max(...predictions));
      const confidence = predictions[maxIndex];
      
      const priceRanges = {
        'Low': { min: 195000, max: 796666 },
        'Mid': { min: 796667, max: 1398333 },
        'High': { min: 1398334, max: 2000000 }
      };
      
      const result = {
        predicted_class: classes[maxIndex],
        confidence: confidence,
        price_range: priceRanges[classes[maxIndex]],
        all_predictions: predictions.map((pred, idx) => ({
          class: classes[idx],
          confidence: pred
        }))
      };
      
      setPrediction(result);
      
    } catch (error) {
      console.error('Error classifying image:', error);
      setError('Failed to classify image. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleDataChange = (field, value) => {
    setDataInputs(prev => ({
      ...prev,
      [field]: value
    }));
  };

  const predictWithBothMethods = async () => {
    if (!selectedImage || !dataInputs.sqft || !dataInputs.bed || !dataInputs.bath || !dataInputs.city) {
      setError('Please fill in all data fields and upload an image.');
      return;
    }
    
    try {
      setLoading(true);
      setError(null);
      
      // First classify the image
      await classifyImage();
      
      // Then get data-based prediction
      const formData = new FormData();
      formData.append('image', selectedImage);
      formData.append('sqft', dataInputs.sqft);
      formData.append('bed', dataInputs.bed);
      formData.append('bath', dataInputs.bath);
      formData.append('city', dataInputs.city);
      
      const response = await axios.post(`${API}/predict-with-image`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });
      
      setCombinedPrediction(response.data);
      
    } catch (error) {
      console.error('Error in combined prediction:', error);
      setError('Failed to get combined prediction. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const getPriceRangeColor = (range) => {
    switch (range) {
      case 'Low': return 'text-green-600 bg-green-50';
      case 'Mid': return 'text-yellow-600 bg-yellow-50';
      case 'High': return 'text-red-600 bg-red-50';
      default: return 'text-gray-600 bg-gray-50';
    }
  };

  const formatPrice = (price) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0
    }).format(price);
  };

  return (
    <div className="min-h-screen bg-gray-50 py-8">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-4">House Price Image Classifier</h1>
          <p className="text-xl text-gray-600 max-w-3xl mx-auto mb-4">
            This section demonstrates cutting-edge machine learning by analyzing house images to predict price ranges. 
            It showcases how AI can extract valuable insights from visual data in real estate markets.
          </p>
          <p className="text-lg text-gray-500 max-w-3xl mx-auto">
            Upload a house image and provide property details to get both AI-powered image classification
            and data-driven price predictions
          </p>
        </div>

        {/* Model Loading Status */}
        {modelLoading && (
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 mb-6">
            <div className="flex items-center space-x-2">
              <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-600"></div>
              <span className="text-blue-600">Loading ML model...</span>
            </div>
          </div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Image Upload Section */}
          <div className="bg-white rounded-xl shadow-sm p-6">
            <div className="flex items-center space-x-2 mb-4">
              <Camera size={24} className="text-blue-600" />
              <h2 className="text-2xl font-bold text-gray-900">Image Upload</h2>
            </div>
            
            <div className="space-y-4">
              <div 
                className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center hover:border-blue-400 transition-colors cursor-pointer"
                onClick={() => fileInputRef.current?.click()}
              >
                {imagePreview ? (
                  <div className="space-y-4">
                    <img 
                      src={imagePreview} 
                      alt="Selected house"
                      className="max-w-full h-48 object-cover rounded-lg mx-auto"
                    />
                    <p className="text-sm text-gray-600">Click to change image</p>
                  </div>
                ) : (
                  <div className="space-y-4">
                    <Upload size={48} className="text-gray-400 mx-auto" />
                    <div>
                      <p className="text-lg font-medium text-gray-900">Upload a house image</p>
                      <p className="text-sm text-gray-600">PNG, JPG, or JPEG files</p>
                    </div>
                  </div>
                )}
              </div>
              
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                onChange={handleImageUpload}
                className="hidden"
              />
              
              <button
                onClick={classifyImage}
                disabled={!selectedImage || !model || loading}
                className="w-full bg-blue-600 text-white py-3 px-4 rounded-lg font-semibold hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors flex items-center justify-center space-x-2"
              >
                {loading ? (
                  <>
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
                    <span>Classifying...</span>
                  </>
                ) : (
                  <>
                    <Zap size={20} />
                    <span>Classify Image</span>
                  </>
                )}
              </button>
            </div>
          </div>

          {/* Property Details Section */}
          <div className="bg-white rounded-xl shadow-sm p-6">
            <div className="flex items-center space-x-2 mb-4">
              <Info size={24} className="text-green-600" />
              <h2 className="text-2xl font-bold text-gray-900">Property Details</h2>
            </div>
            
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Square Footage</label>
                <input
                  type="number"
                  value={dataInputs.sqft}
                  onChange={(e) => handleDataChange('sqft', e.target.value)}
                  placeholder="e.g., 1500"
                  className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                />
              </div>
              
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Bedrooms</label>
                  <input
                    type="number"
                    value={dataInputs.bed}
                    onChange={(e) => handleDataChange('bed', e.target.value)}
                    placeholder="e.g., 3"
                    className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  />
                </div>
                
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Bathrooms</label>
                  <input
                    type="number"
                    step="0.5"
                    value={dataInputs.bath}
                    onChange={(e) => handleDataChange('bath', e.target.value)}
                    placeholder="e.g., 2"
                    className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  />
                </div>
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">City</label>
                <input
                  type="text"
                  value={dataInputs.city}
                  onChange={(e) => handleDataChange('city', e.target.value)}
                  placeholder="e.g., Los Angeles, CA"
                  className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                />
              </div>
              
              <button
                onClick={predictWithBothMethods}
                disabled={!selectedImage || !model || loading || !dataInputs.sqft || !dataInputs.bed || !dataInputs.bath || !dataInputs.city}
                className="w-full bg-green-600 text-white py-3 px-4 rounded-lg font-semibold hover:bg-green-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors flex items-center justify-center space-x-2"
              >
                {loading ? (
                  <>
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
                    <span>Predicting...</span>
                  </>
                ) : (
                  <>
                    <Zap size={20} />
                    <span>Get Combined Prediction</span>
                  </>
                )}
              </button>
            </div>
          </div>
        </div>

        {/* Error Display */}
        {error && (
          <div className="mt-6 bg-red-50 border border-red-200 rounded-lg p-4">
            <div className="flex items-center space-x-2">
              <AlertCircle size={20} className="text-red-600" />
              <span className="text-red-800">{error}</span>
            </div>
          </div>
        )}

        {/* Prediction Results */}
        {prediction && (
          <div className="mt-8 bg-white rounded-xl shadow-sm p-6">
            <div className="flex items-center space-x-2 mb-4">
              <CheckCircle size={24} className="text-green-600" />
              <h2 className="text-2xl font-bold text-gray-900">Image Classification Result</h2>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="space-y-4">
                <div className={`p-4 rounded-lg ${getPriceRangeColor(prediction.predicted_class)}`}>
                  <p className="text-sm font-medium">Predicted Price Range</p>
                  <p className="text-2xl font-bold">{prediction.predicted_class}</p>
                  <p className="text-sm">
                    {formatPrice(prediction.price_range.min)} - {formatPrice(prediction.price_range.max)}
                  </p>
                </div>
                
                <div className="bg-gray-50 p-4 rounded-lg">
                  <p className="text-sm font-medium text-gray-700">Confidence</p>
                  <p className="text-xl font-bold text-gray-900">{(prediction.confidence * 100).toFixed(1)}%</p>
                </div>
              </div>
              
              <div className="space-y-3">
                <p className="text-sm font-medium text-gray-700">All Predictions</p>
                {prediction.all_predictions.map((pred, index) => (
                  <div key={index} className="flex justify-between items-center p-2 bg-gray-50 rounded">
                    <span className="font-medium">{pred.class}</span>
                    <span className="text-sm text-gray-600">{(pred.confidence * 100).toFixed(1)}%</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Combined Prediction Results */}
        {combinedPrediction && (
          <div className="mt-8 bg-white rounded-xl shadow-sm p-6">
            <div className="flex items-center space-x-2 mb-4">
              <CheckCircle size={24} className="text-blue-600" />
              <h2 className="text-2xl font-bold text-gray-900">Combined Prediction Analysis</h2>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="space-y-4">
                <h3 className="text-lg font-semibold text-gray-900">Data-Based Prediction</h3>
                <div className="bg-blue-50 p-4 rounded-lg">
                  <p className="text-sm font-medium text-blue-700">Predicted Price</p>
                  <p className="text-2xl font-bold text-blue-900">
                    {formatPrice(combinedPrediction.data_prediction.predicted_price)}
                  </p>
                  <p className="text-sm text-blue-600">
                    Range: {combinedPrediction.data_prediction.price_range}
                  </p>
                </div>
                
                <div className="bg-gray-50 p-4 rounded-lg">
                  <p className="text-sm font-medium text-gray-700">Confidence</p>
                  <p className="text-lg font-bold text-gray-900">
                    {(combinedPrediction.data_prediction.confidence * 100).toFixed(1)}%
                  </p>
                </div>
              </div>
              
              <div className="space-y-4">
                <h3 className="text-lg font-semibold text-gray-900">Key Factors</h3>
                <div className="space-y-2">
                  {Object.entries(combinedPrediction.data_prediction.factors).map(([key, value]) => (
                    <div key={key} className="text-sm">
                      <span className="font-medium text-gray-700">{key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}: </span>
                      <span className="text-gray-600">{value}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Information Section */}
        <div className="mt-8 bg-blue-50 border border-blue-200 rounded-lg p-6">
          <h3 className="text-lg font-semibold text-blue-900 mb-3">How It Works</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm text-blue-800">
            <div>
              <p className="font-medium mb-2">🤖 Image Classification:</p>
              <p>Uses Google Teachable Machine model trained on 105 randomly selected house images across three price categories.</p>
            </div>
            <div>
              <p className="font-medium mb-2">📊 Data Analysis:</p>
              <p>Leverages statistical insights with square footage as the strongest predictor (0.58 correlation).</p>
            </div>
          </div>
        </div>

        {/* In-Depth AI Model Explanation - Collapsible */}
        <div className="mt-8 bg-white rounded-xl shadow-sm">
          <div 
            className="flex items-center justify-between p-6 cursor-pointer hover:bg-gray-50 transition-colors"
            onClick={() => setIsAIExplanationOpen(!isAIExplanationOpen)}
          >
            <div className="flex items-center space-x-2">
              <Brain size={24} className="text-purple-600" />
              <h2 className="text-2xl font-bold text-gray-900">In-Depth: How Our Dual AI House Price Prediction System Works</h2>
            </div>
            {isAIExplanationOpen ? (
              <ChevronUp size={24} className="text-gray-400" />
            ) : (
              <ChevronDown size={24} className="text-gray-400" />
            )}
          </div>
          
          {isAIExplanationOpen && (
            <div className="px-6 pb-6 space-y-8">
              {/* Overview Section */}
              <div className="bg-gradient-to-r from-purple-50 to-blue-50 p-6 rounded-lg">
                <h3 className="text-xl font-semibold text-gray-900 mb-3">🤖 Dual AI Approach Overview</h3>
                <p className="text-gray-700 mb-4">
                  Our system combines <strong>two different AI technologies</strong> to provide comprehensive house price predictions:
                </p>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="bg-white p-4 rounded-lg">
                    <h4 className="font-semibold text-purple-900 mb-2">📸 Image Classification AI</h4>
                    <p className="text-sm text-gray-700">Deep learning CNN that analyzes house photos to predict price categories</p>
                  </div>
                  <div className="bg-white p-4 rounded-lg">
                    <h4 className="font-semibold text-blue-900 mb-2">📊 Data-Driven Prediction AI</h4>
                    <p className="text-sm text-gray-700">Statistical ML model using property features for precise price estimation</p>
                  </div>
                </div>
              </div>

              {/* Data-Driven AI Section */}
              <div className="border-l-4 border-blue-500 pl-6">
                <h3 className="text-xl font-semibold text-gray-900 mb-3">📊 Data-Driven Price Prediction AI</h3>
                <div className="space-y-4 text-gray-700">
                  <p>
                    <strong>How it works:</strong> When you enter property details (square footage, bedrooms, bathrooms, city), 
                    our statistical machine learning model analyzes these features against our database of 15,474 Southern California homes 
                    to predict an exact price.
                  </p>
                  
                  <div className="bg-blue-50 p-4 rounded-lg">
                    <h4 className="font-semibold text-blue-900 mb-2">🧮 Algorithm Components:</h4>
                    <ul className="space-y-1 text-sm">
                      <li><strong>Primary Factor:</strong> Square footage (0.583 correlation coefficient - our strongest predictor)</li>
                      <li><strong>Location Adjustments:</strong> City-specific price multipliers based on historical market data</li>
                      <li><strong>Feature Modifiers:</strong> Bedroom and bathroom count adjustments</li>
                      <li><strong>Edge Case Handling:</strong> Price caps, minimum thresholds, and unknown city fallbacks</li>
                      <li><strong>Confidence Scoring:</strong> Reliability metrics based on data availability and feature completeness</li>
                    </ul>
                  </div>
                  
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div>
                      <h4 className="font-semibold text-blue-900 mb-2">🎯 Prediction Process:</h4>
                      <ol className="text-sm space-y-1 list-decimal list-inside">
                        <li>Base price calculated from square footage</li>
                        <li>City-specific adjustment applied</li>
                        <li>Bedroom/bathroom modifications</li>
                        <li>Market trend considerations</li>
                        <li>Final price with confidence score</li>
                      </ol>
                    </div>
                    <div>
                      <h4 className="font-semibold text-blue-900 mb-2">📈 Model Performance:</h4>
                      <ul className="text-sm space-y-1">
                        <li>Trained on 15,474 real property records</li>
                        <li>Covers 415+ Southern California cities</li>
                        <li>Price range: $195K - $2M+</li>
                        <li>Correlation strength: Strong (0.583)</li>
                      </ul>
                    </div>
                  </div>
                </div>
              </div>

              {/* Image Classification AI Section */}
              <div className="border-l-4 border-purple-500 pl-6">
                <h3 className="text-xl font-semibold text-gray-900 mb-3">📸 Image Classification AI</h3>
                <div className="space-y-3 text-gray-700">
                  <p>
                    <strong>Model Type:</strong> Our image classifier uses a <strong>Convolutional Neural Network (CNN)</strong> 
                    implemented through Google's Teachable Machine platform. CNNs are specifically designed for image recognition 
                    tasks, using layers of convolution and pooling operations to detect visual patterns and features.
                  </p>
                  <p>
                    <strong>Architecture:</strong> The model employs transfer learning, built on a pre-trained MobileNet backbone 
                    that has been fine-tuned on our specific house image dataset. This approach leverages existing knowledge 
                    from millions of images while adapting to our house price prediction task.
                  </p>
                </div>
              </div>

              {/* Data Preprocessing Section */}
              <div className="border-l-4 border-green-500 pl-6">
                <h3 className="text-xl font-semibold text-gray-900 mb-3">⚙️ Data Preprocessing Pipeline</h3>
                <div className="space-y-3 text-gray-700">
                  <p>
                    <strong>Image Processing:</strong> All uploaded images undergo standardized preprocessing:
                  </p>
                  <ul className="list-disc pl-6 space-y-1">
                    <li><strong>Resizing:</strong> Images are automatically resized to 224×224 pixels for consistency</li>
                    <li><strong>Color Normalization:</strong> RGB values are normalized to a 0-1 scale for optimal neural network performance</li>
                    <li><strong>Format Conversion:</strong> All images are converted to RGB format, regardless of original format</li>
                    <li><strong>Tensor Transformation:</strong> Images are converted to numerical tensors that the model can process</li>
                  </ul>
                  <p>
                    <strong>Training Data:</strong> The model was trained on 105 carefully selected house images from our dataset, 
                    representing diverse architectural styles, conditions, and price ranges across Southern California.
                  </p>
                </div>
              </div>

              {/* Combined Approach Section */}
              <div className="border-l-4 border-orange-500 pl-6">
                <h3 className="text-xl font-semibold text-gray-900 mb-3">🔄 Combined Prediction Power</h3>
                <div className="space-y-3 text-gray-700">
                  <p>
                    <strong>Best of Both Worlds:</strong> When you use both systems together (upload an image AND provide property details), 
                    you get complementary insights:
                  </p>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="bg-purple-50 p-4 rounded-lg">
                      <h4 className="font-semibold text-purple-900 mb-2">Visual Analysis</h4>
                      <p className="text-sm">Image AI captures aesthetic factors, condition, architectural style, and curb appeal</p>
                    </div>
                    <div className="bg-blue-50 p-4 rounded-lg">
                      <h4 className="font-semibold text-blue-900 mb-2">Quantitative Analysis</h4>
                      <p className="text-sm">Data AI provides precise calculations based on measurable property characteristics</p>
                    </div>
                  </div>
                </div>
              </div>

              {/* Technical Note */}
              <div className="p-4 bg-gray-50 rounded-lg">
                <h4 className="font-semibold text-gray-900 mb-2">🔬 Technical Implementation Note</h4>
                <p className="text-sm text-gray-700">
                  This dual-approach system demonstrates the practical application of both computer vision and statistical machine learning 
                  in real estate valuation. The combination provides more comprehensive and reliable price predictions than either method alone.
                </p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default ImageClassifier;