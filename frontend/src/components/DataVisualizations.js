import React, { useState, useEffect } from "react";
import { BarChart3, PieChart, TrendingUp, Info, Database } from "lucide-react";
import axios from "axios";

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const DataVisualizations = () => {
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchStats();
  }, []);

  const fetchStats = async () => {
    try {
      const response = await axios.get(`${API}/stats`);
      setStats(response.data);
    } catch (error) {
      console.error('Error fetching stats:', error);
      setError('Failed to load statistics');
    } finally {
      setLoading(false);
    }
  };

  const formatNumber = (num) => {
    return new Intl.NumberFormat('en-US').format(num);
  };

  const formatPrice = (price) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0
    }).format(price);
  };

  const getPercentage = (value, total) => {
    return ((value / total) * 100).toFixed(1);
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 py-8">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
            <p className="text-gray-600">Loading visualizations...</p>
          </div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-gray-50 py-8">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center">
            <div className="bg-red-50 border border-red-200 rounded-lg p-6">
              <p className="text-red-800">{error}</p>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50 py-8">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-4">Data Visualizations</h1>
          <p className="text-xl text-gray-600 max-w-3xl mx-auto mb-4">
            This section presents comprehensive visual analysis of our Southern California housing dataset. 
            Understanding these patterns helps reveal market trends and pricing factors that drive our predictive models.
          </p>
          <p className="text-lg text-gray-500 max-w-3xl mx-auto">
            Explore charts showing price distributions, property features, and market patterns
            across {stats?.total_houses?.toLocaleString()} Southern California homes
          </p>
        </div>

        {/* Overview Stats */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
          <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-200">
            <div className="flex items-center space-x-2 mb-2">
              <Database size={20} className="text-blue-600" />
              <p className="text-sm text-gray-600">Total Properties</p>
            </div>
            <p className="text-2xl font-bold text-gray-900">{formatNumber(stats?.total_houses || 0)}</p>
          </div>
          <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-200">
            <div className="flex items-center space-x-2 mb-2">
              <TrendingUp size={20} className="text-green-600" />
              <p className="text-sm text-gray-600">Average Price</p>
            </div>
            <p className="text-2xl font-bold text-gray-900">{formatPrice(stats?.avg_price || 0)}</p>
          </div>
          <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-200">
            <div className="flex items-center space-x-2 mb-2">
              <BarChart3 size={20} className="text-purple-600" />
              <p className="text-sm text-gray-600">Average Sqft</p>
            </div>
            <p className="text-2xl font-bold text-gray-900">{formatNumber(stats?.avg_sqft || 0)}</p>
          </div>
          <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-200">
            <div className="flex items-center space-x-2 mb-2">
              <PieChart size={20} className="text-red-600" />
              <p className="text-sm text-gray-600">Avg Bed/Bath</p>
            </div>
            <p className="text-2xl font-bold text-gray-900">{stats?.avg_bed?.toFixed(1) || 0} / {stats?.avg_bath?.toFixed(1) || 0}</p>
          </div>
        </div>

        {/* Price Distribution */}
        <div className="bg-white rounded-xl shadow-sm p-6 mb-8">
          <div className="flex items-center space-x-2 mb-6">
            <TrendingUp size={24} className="text-blue-600" />
            <h2 className="text-2xl font-bold text-gray-900">Price Distribution</h2>
          </div>
          
          <div className="mb-6">
            <img 
              src="/house prices historgram.png" 
              alt="House Price Distribution Histogram"
              className="w-full max-w-4xl mx-auto rounded-lg shadow-sm"
              onError={(e) => {
                e.target.style.display = 'none';
                e.target.nextSibling.style.display = 'block';
              }}
            />
            <div className="text-center text-gray-500 mt-4" style={{display: 'none'}}>
              House Price Distribution Histogram
            </div>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="bg-green-50 p-4 rounded-lg">
              <div className="text-center">
                <p className="text-sm font-medium text-green-700">Low Range</p>
                <p className="text-lg font-bold text-green-900">
                  {formatPrice(stats?.price_ranges?.low?.min || 0)} - {formatPrice(stats?.price_ranges?.low?.max || 0)}
                </p>
                <p className="text-sm text-green-600">
                  {formatNumber(stats?.price_ranges?.low?.count || 0)} properties
                </p>
                <p className="text-xs text-green-500">
                  {getPercentage(stats?.price_ranges?.low?.count || 0, stats?.total_houses || 1)}% of market
                </p>
              </div>
            </div>
            <div className="bg-yellow-50 p-4 rounded-lg">
              <div className="text-center">
                <p className="text-sm font-medium text-yellow-700">Mid Range</p>
                <p className="text-lg font-bold text-yellow-900">
                  {formatPrice(stats?.price_ranges?.mid?.min || 0)} - {formatPrice(stats?.price_ranges?.mid?.max || 0)}
                </p>
                <p className="text-sm text-yellow-600">
                  {formatNumber(stats?.price_ranges?.mid?.count || 0)} properties
                </p>
                <p className="text-xs text-yellow-500">
                  {getPercentage(stats?.price_ranges?.mid?.count || 0, stats?.total_houses || 1)}% of market
                </p>
              </div>
            </div>
            <div className="bg-red-50 p-4 rounded-lg">
              <div className="text-center">
                <p className="text-sm font-medium text-red-700">High Range</p>
                <p className="text-lg font-bold text-red-900">
                  {formatPrice(stats?.price_ranges?.high?.min || 0)} - {formatPrice(stats?.price_ranges?.high?.max || 0)}
                </p>
                <p className="text-sm text-red-600">
                  {formatNumber(stats?.price_ranges?.high?.count || 0)} properties
                </p>
                <p className="text-xs text-red-500">
                  {getPercentage(stats?.price_ranges?.high?.count || 0, stats?.total_houses || 1)}% of market
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* Square Footage Distribution */}
        <div className="bg-white rounded-xl shadow-sm p-6 mb-8">
          <div className="flex items-center space-x-2 mb-6">
            <BarChart3 size={24} className="text-purple-600" />
            <h2 className="text-2xl font-bold text-gray-900">Square Footage Distribution</h2>
          </div>
          
          <div className="mb-4">
            <img 
              src="/house sqft histogram.png" 
              alt="House Square Footage Distribution"
              className="w-full max-w-4xl mx-auto rounded-lg shadow-sm"
              onError={(e) => {
                e.target.style.display = 'none';
                e.target.nextSibling.style.display = 'block';
              }}
            />
            <div className="text-center text-gray-500 mt-4" style={{display: 'none'}}>
              House Square Footage Distribution
            </div>
          </div>
          
          <div className="bg-purple-50 p-4 rounded-lg">
            <div className="text-center">
              <p className="text-sm font-medium text-purple-700">Key Insight</p>
              <p className="text-lg font-bold text-purple-900">
                Average: {formatNumber(stats?.avg_sqft || 0)} sqft
              </p>
              <p className="text-sm text-purple-600">
                Most homes range from 1,000 to 3,000 square feet, with the majority clustered around the average
              </p>
            </div>
          </div>
        </div>

        {/* Bedrooms Distribution */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-8">
          <div className="bg-white rounded-xl shadow-sm p-6">
            <div className="flex items-center space-x-2 mb-6">
              <PieChart size={24} className="text-green-600" />
              <h2 className="text-2xl font-bold text-gray-900">Bedrooms Distribution</h2>
            </div>
            
            <div className="mb-4">
              <img 
                src="/bedroom pie chart.png" 
                alt="Bedrooms Distribution Pie Chart"
                className="w-full max-w-md mx-auto rounded-lg shadow-sm"
                onError={(e) => {
                  e.target.style.display = 'none';
                  e.target.nextSibling.style.display = 'block';
                }}
              />
              <div className="text-center text-gray-500 mt-4" style={{display: 'none'}}>
                Bedrooms Distribution Pie Chart
              </div>
            </div>
            
            <div className="space-y-2">
              {Object.entries(stats?.bed_distribution || {}).map(([beds, count]) => (
                <div key={beds} className="flex justify-between items-center p-2 bg-gray-50 rounded">
                  <span className="font-medium">{beds} bedroom{beds !== '1' ? 's' : ''}</span>
                  <div className="text-right">
                    <span className="text-sm text-gray-600">{formatNumber(count)}</span>
                    <span className="text-xs text-gray-500 ml-2">
                      ({getPercentage(count, stats?.total_houses || 1)}%)
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="bg-white rounded-xl shadow-sm p-6">
            <div className="flex items-center space-x-2 mb-6">
              <PieChart size={24} className="text-blue-600" />
              <h2 className="text-2xl font-bold text-gray-900">Bathrooms Distribution</h2>
            </div>
            
            <div className="mb-4 p-3 bg-blue-50 border border-blue-200 rounded-lg">
              <div className="flex items-start space-x-2">
                <Info size={16} className="text-blue-600 mt-0.5" />
                <div className="text-sm text-blue-800">
                  <p className="font-medium mb-1">Understanding Bathroom Numbers</p>
                  <p>For all the numbers shown here, the first number refers to full bathrooms, and the second refers to half bathrooms. So if a value is 5.2, it means 5 full baths and 2 half baths.</p>
                </div>
              </div>
            </div>
            
            <div className="mb-4">
              <img 
                src="/bthrm pie chart.png" 
                alt="Bathrooms Distribution Pie Chart"
                className="w-full max-w-md mx-auto rounded-lg shadow-sm"
                onError={(e) => {
                  e.target.style.display = 'none';
                  e.target.nextSibling.style.display = 'block';
                }}
              />
              <div className="text-center text-gray-500 mt-4" style={{display: 'none'}}>
                Bathrooms Distribution Pie Chart
              </div>
            </div>
            
            <div className="space-y-2">
              {Object.entries(stats?.bath_distribution || {}).map(([baths, count]) => (
                <div key={baths} className="flex justify-between items-center p-2 bg-gray-50 rounded">
                  <span className="font-medium">{baths} bathroom{baths !== '1' ? 's' : ''}</span>
                  <div className="text-right">
                    <span className="text-sm text-gray-600">{formatNumber(count)}</span>
                    <span className="text-xs text-gray-500 ml-2">
                      ({getPercentage(count, stats?.total_houses || 1)}%)
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Key Insights */}
        <div className="bg-blue-50 border border-blue-200 rounded-xl p-6">
          <div className="flex items-center space-x-2 mb-4">
            <Info size={24} className="text-blue-600" />
            <h2 className="text-2xl font-bold text-blue-900">Key Insights</h2>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <h3 className="font-semibold text-blue-900 mb-2">Price Distribution</h3>
              <ul className="space-y-1 text-blue-800">
                <li>• Most homes (≈{getPercentage(stats?.price_ranges?.low?.count || 0, stats?.total_houses || 1)}%) are in the lower price range</li>
                <li>• Average price: {formatPrice(stats?.avg_price || 0)}</li>
                <li>• Wide price variation reflects diverse market segments</li>
              </ul>
            </div>
            
            <div>
              <h3 className="font-semibold text-blue-900 mb-2">Property Features</h3>
              <ul className="space-y-1 text-blue-800">
                <li>• Most homes have 3-4 bedrooms</li>
                <li>• 2-3 bathrooms are most common</li>
                <li>• Average size: {formatNumber(stats?.avg_sqft || 0)} sqft</li>
                <li>• Property sizes vary widely across the region</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default DataVisualizations;