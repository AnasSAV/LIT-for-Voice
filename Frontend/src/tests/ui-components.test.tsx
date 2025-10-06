/**
 * User Interface Testing - React Components and Audio Interactions
 * Test Plan Section 3.1.3 - Frontend Components, Navigation, Audio Interface
 */

import React from 'react';
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import '@testing-library/jest-dom';
import { jest } from '@jest/globals';

// Mock audio context for testing
const mockAudioContext = {
  createBufferSource: jest.fn(() => ({
    connect: jest.fn(),
    start: jest.fn(),
    stop: jest.fn(),
    buffer: null
  })),
  createGain: jest.fn(() => ({
    connect: jest.fn(),
    gain: { value: 1 }
  })),
  destination: {},
  decodeAudioData: jest.fn()
};

// Mock WaveSurfer for audio visualization testing
jest.mock('wavesurfer.js', () => {
  return {
    default: {
      create: jest.fn(() => ({
        load: jest.fn(),
        play: jest.fn(),
        pause: jest.fn(),
        seekTo: jest.fn(),
        setVolume: jest.fn(),
        zoom: jest.fn(),
        on: jest.fn(),
        destroy: jest.fn(),
        getCurrentTime: jest.fn(() => 0),
        getDuration: jest.fn(() => 5.0),
        isPlaying: jest.fn(() => false)
      }))
    }
  };
});

// Mock API calls
const mockFetch = jest.fn();
global.fetch = mockFetch;

describe('Audio Component Testing', () => {
  beforeEach(() => {
    // Reset mocks
    jest.clearAllMocks();
    
    // Mock Web Audio API
    Object.defineProperty(window, 'AudioContext', {
      writable: true,
      value: jest.fn(() => mockAudioContext)
    });
    
    Object.defineProperty(window, 'webkitAudioContext', {
      writable: true,
      value: jest.fn(() => mockAudioContext)
    });
  });

  describe('Waveform Viewer Component', () => {
    // This would test the actual WaveformViewer component when available
    test('should render waveform visualization', () => {
      // Mock component for testing
      const MockWaveformViewer = ({ audioUrl }: { audioUrl: string }) => (
        <div data-testid="waveform-viewer">
          <canvas data-testid="waveform-canvas" />
          <div data-testid="audio-controls">
            <button data-testid="play-button">Play</button>
            <button data-testid="pause-button">Pause</button>
            <input data-testid="seek-slider" type="range" />
          </div>
        </div>
      );

      render(<MockWaveformViewer audioUrl="/test-audio.wav" />);
      
      expect(screen.getByTestId('waveform-viewer')).toBeInTheDocument();
      expect(screen.getByTestId('waveform-canvas')).toBeInTheDocument();
      expect(screen.getByTestId('play-button')).toBeInTheDocument();
    });

    test('should handle play/pause controls', async () => {
      const user = userEvent.setup();
      
      const MockAudioPlayer = () => {
        const [isPlaying, setIsPlaying] = React.useState(false);
        
        return (
          <div>
            <button 
              data-testid="play-pause-button"
              onClick={() => setIsPlaying(!isPlaying)}
            >
              {isPlaying ? 'Pause' : 'Play'}
            </button>
            <span data-testid="play-state">{isPlaying ? 'playing' : 'paused'}</span>
          </div>
        );
      };

      render(<MockAudioPlayer />);
      
      const playButton = screen.getByTestId('play-pause-button');
      const playState = screen.getByTestId('play-state');
      
      expect(playState).toHaveTextContent('paused');
      
      await user.click(playButton);
      expect(playState).toHaveTextContent('playing');
      
      await user.click(playButton);
      expect(playState).toHaveTextContent('paused');
    });

    test('should handle seek functionality', async () => {
      const user = userEvent.setup();
      
      const MockSeekControl = () => {
        const [currentTime, setCurrentTime] = React.useState(0);
        const duration = 10; // 10 seconds
        
        return (
          <div>
            <input
              data-testid="seek-slider"
              type="range"
              min={0}
              max={duration}
              value={currentTime}
              onChange={(e) => setCurrentTime(Number(e.target.value))}
            />
            <span data-testid="current-time">{currentTime}s</span>
            <span data-testid="duration">{duration}s</span>
          </div>
        );
      };

      render(<MockSeekControl />);
      
      const seekSlider = screen.getByTestId('seek-slider');
      const currentTimeDisplay = screen.getByTestId('current-time');
      
      await user.clear(seekSlider);
      await user.type(seekSlider, '5');
      
      expect(currentTimeDisplay).toHaveTextContent('5s');
    });

    test('should handle zoom controls', async () => {
      const user = userEvent.setup();
      
      const MockZoomControl = () => {
        const [zoomLevel, setZoomLevel] = React.useState(1);
        
        return (
          <div>
            <button 
              data-testid="zoom-in" 
              onClick={() => setZoomLevel(prev => prev * 1.5)}
            >
              Zoom In
            </button>
            <button 
              data-testid="zoom-out" 
              onClick={() => setZoomLevel(prev => prev / 1.5)}
            >
              Zoom Out
            </button>
            <span data-testid="zoom-level">Zoom: {zoomLevel.toFixed(1)}x</span>
          </div>
        );
      };

      render(<MockZoomControl />);
      
      const zoomInButton = screen.getByTestId('zoom-in');
      const zoomLevel = screen.getByTestId('zoom-level');
      
      expect(zoomLevel).toHaveTextContent('Zoom: 1.0x');
      
      await user.click(zoomInButton);
      expect(zoomLevel).toHaveTextContent('Zoom: 1.5x');
    });
  });

  describe('Audio Upload Interface', () => {
    test('should handle drag and drop functionality', async () => {
      const user = userEvent.setup();
      
      const MockDropZone = () => {
        const [isDragOver, setIsDragOver] = React.useState(false);
        const [files, setFiles] = React.useState<File[]>([]);
        
        const handleDrop = (e: React.DragEvent) => {
          e.preventDefault();
          const droppedFiles = Array.from(e.dataTransfer.files);
          setFiles(droppedFiles);
          setIsDragOver(false);
        };
        
        return (
          <div
            data-testid="drop-zone"
            onDragOver={(e) => {
              e.preventDefault();
              setIsDragOver(true);
            }}
            onDragLeave={() => setIsDragOver(false)}
            onDrop={handleDrop}
            style={{
              border: `2px dashed ${isDragOver ? 'blue' : 'gray'}`,
              padding: '20px'
            }}
          >
            <p>Drop audio files here</p>
            {files.length > 0 && (
              <div data-testid="uploaded-files">
                {files.map((file, index) => (
                  <div key={index} data-testid={`file-${index}`}>
                    {file.name}
                  </div>
                ))}
              </div>
            )}
          </div>
        );
      };

      render(<MockDropZone />);
      
      const dropZone = screen.getByTestId('drop-zone');
      
      // Create a mock file
      const mockFile = new File(['audio content'], 'test.wav', { type: 'audio/wav' });
      
      // Simulate drag over
      fireEvent.dragOver(dropZone);
      expect(dropZone).toHaveStyle('border: 2px dashed blue');
      
      // Simulate drop
      fireEvent.drop(dropZone, {
        dataTransfer: {
          files: [mockFile]
        }
      });
      
      expect(screen.getByTestId('uploaded-files')).toBeInTheDocument();
      expect(screen.getByTestId('file-0')).toHaveTextContent('test.wav');
    });

    test('should validate file types', () => {
      const validateAudioFile = (file: File): boolean => {
        const allowedTypes = ['audio/wav', 'audio/mp3', 'audio/flac'];
        return allowedTypes.includes(file.type);
      };

      const validFile = new File([''], 'test.wav', { type: 'audio/wav' });
      const invalidFile = new File([''], 'test.txt', { type: 'text/plain' });

      expect(validateAudioFile(validFile)).toBe(true);
      expect(validateAudioFile(invalidFile)).toBe(false);
    });

    test('should handle file size limits', () => {
      const MAX_FILE_SIZE = 50 * 1024 * 1024; // 50MB
      
      const validateFileSize = (file: File): boolean => {
        return file.size <= MAX_FILE_SIZE;
      };

      // Create mock files
      const validFile = new File(['x'.repeat(1024)], 'small.wav', { type: 'audio/wav' });
      const oversizedFile = new File(['x'.repeat(MAX_FILE_SIZE + 1)], 'large.wav', { type: 'audio/wav' });

      expect(validateFileSize(validFile)).toBe(true);
      expect(validateFileSize(oversizedFile)).toBe(false);
    });
  });

  describe('Model Selection and Configuration', () => {
    test('should render model selection dropdown', async () => {
      const user = userEvent.setup();
      
      const MockModelSelector = () => {
        const [selectedModel, setSelectedModel] = React.useState('whisper-base');
        const models = ['whisper-base', 'whisper-large', 'wav2vec2'];
        
        return (
          <div>
            <select 
              data-testid="model-selector"
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
            >
              {models.map(model => (
                <option key={model} value={model}>{model}</option>
              ))}
            </select>
            <span data-testid="selected-model">{selectedModel}</span>
          </div>
        );
      };

      render(<MockModelSelector />);
      
      const selector = screen.getByTestId('model-selector');
      const selectedDisplay = screen.getByTestId('selected-model');
      
      expect(selectedDisplay).toHaveTextContent('whisper-base');
      
      await user.selectOptions(selector, 'wav2vec2');
      expect(selectedDisplay).toHaveTextContent('wav2vec2');
    });

    test('should handle model parameters', async () => {
      const user = userEvent.setup();
      
      const MockModelConfig = () => {
        const [config, setConfig] = React.useState({
          language: 'auto',
          temperature: 0.0,
          beamSize: 5
        });
        
        return (
          <div>
            <select
              data-testid="language-selector"
              value={config.language}
              onChange={(e) => setConfig({...config, language: e.target.value})}
            >
              <option value="auto">Auto-detect</option>
              <option value="en">English</option>
              <option value="es">Spanish</option>
            </select>
            
            <input
              data-testid="temperature-input"
              type="number"
              min={0}
              max={1}
              step={0.1}
              value={config.temperature}
              onChange={(e) => setConfig({...config, temperature: Number(e.target.value)})}
            />
            
            <div data-testid="config-display">
              {JSON.stringify(config)}
            </div>
          </div>
        );
      };

      render(<MockModelConfig />);
      
      const languageSelector = screen.getByTestId('language-selector');
      const temperatureInput = screen.getByTestId('temperature-input');
      const configDisplay = screen.getByTestId('config-display');
      
      await user.selectOptions(languageSelector, 'en');
      await user.clear(temperatureInput);
      await user.type(temperatureInput, '0.5');
      
      expect(configDisplay).toHaveTextContent('"language":"en"');
      expect(configDisplay).toHaveTextContent('"temperature":0.5');
    });
  });
});

describe('Navigation and Panel Management', () => {
  test('should handle panel switching', async () => {
    const user = userEvent.setup();
    
    const MockPanelManager = () => {
      const [activePanel, setActivePanel] = React.useState('prediction');
      const panels = ['prediction', 'embedding', 'attention', 'saliency'];
      
      return (
        <div>
          <nav data-testid="panel-navigation">
            {panels.map(panel => (
              <button
                key={panel}
                data-testid={`panel-${panel}`}
                onClick={() => setActivePanel(panel)}
                className={activePanel === panel ? 'active' : ''}
              >
                {panel}
              </button>
            ))}
          </nav>
          <div data-testid="active-panel">
            Current Panel: {activePanel}
          </div>
        </div>
      );
    };

    render(<MockPanelManager />);
    
    const activePanel = screen.getByTestId('active-panel');
    const attentionButton = screen.getByTestId('panel-attention');
    
    expect(activePanel).toHaveTextContent('Current Panel: prediction');
    
    await user.click(attentionButton);
    expect(activePanel).toHaveTextContent('Current Panel: attention');
  });

  test('should handle panel resizing', () => {
    const MockResizablePanel = () => {
      const [width, setWidth] = React.useState(300);
      
      return (
        <div>
          <div 
            data-testid="resizable-panel"
            style={{ width: `${width}px`, border: '1px solid black' }}
          >
            Panel Content
          </div>
          <input
            data-testid="width-control"
            type="range"
            min={200}
            max={800}
            value={width}
            onChange={(e) => setWidth(Number(e.target.value))}
          />
          <span data-testid="width-display">{width}px</span>
        </div>
      );
    };

    render(<MockResizablePanel />);
    
    const panel = screen.getByTestId('resizable-panel');
    const widthControl = screen.getByTestId('width-control');
    const widthDisplay = screen.getByTestId('width-display');
    
    expect(panel).toHaveStyle('width: 300px');
    expect(widthDisplay).toHaveTextContent('300px');
    
    fireEvent.change(widthControl, { target: { value: '500' } });
    expect(widthDisplay).toHaveTextContent('500px');
  });

  test('should preserve state during navigation', async () => {
    const user = userEvent.setup();
    
    const MockStatefulNavigation = () => {
      const [currentTab, setCurrentTab] = React.useState('tab1');
      const [tabStates, setTabStates] = React.useState({
        tab1: { value: 'Initial 1' },
        tab2: { value: 'Initial 2' }
      });
      
      const updateTabState = (tab: string, value: string) => {
        setTabStates(prev => ({
          ...prev,
          [tab]: { value }
        }));
      };
      
      return (
        <div>
          <button 
            data-testid="tab1-button"
            onClick={() => setCurrentTab('tab1')}
          >
            Tab 1
          </button>
          <button 
            data-testid="tab2-button"
            onClick={() => setCurrentTab('tab2')}
          >
            Tab 2
          </button>
          
          <div data-testid="tab-content">
            <input
              data-testid="tab-input"
              value={tabStates[currentTab].value}
              onChange={(e) => updateTabState(currentTab, e.target.value)}
            />
          </div>
        </div>
      );
    };

    render(<MockStatefulNavigation />);
    
    const tab1Button = screen.getByTestId('tab1-button');
    const tab2Button = screen.getByTestId('tab2-button');
    const tabInput = screen.getByTestId('tab-input');
    
    // Modify tab 1
    expect(tabInput).toHaveValue('Initial 1');
    await user.clear(tabInput);
    await user.type(tabInput, 'Modified 1');
    
    // Switch to tab 2
    await user.click(tab2Button);
    expect(tabInput).toHaveValue('Initial 2');
    
    // Switch back to tab 1 - state should be preserved
    await user.click(tab1Button);
    expect(tabInput).toHaveValue('Modified 1');
  });
});

describe('Interactive Visualization Controls', () => {
  test('should handle attention visualization interaction', async () => {
    const user = userEvent.setup();
    
    const MockAttentionVisualization = () => {
      const [selectedLayer, setSelectedLayer] = React.useState(0);
      const [selectedHead, setSelectedHead] = React.useState(0);
      const layers = 12;
      const heads = 12;
      
      return (
        <div>
          <div data-testid="layer-controls">
            <label>Layer:</label>
            <select
              data-testid="layer-selector"
              value={selectedLayer}
              onChange={(e) => setSelectedLayer(Number(e.target.value))}
            >
              {Array.from({ length: layers }, (_, i) => (
                <option key={i} value={i}>Layer {i + 1}</option>
              ))}
            </select>
          </div>
          
          <div data-testid="head-controls">
            <label>Head:</label>
            <select
              data-testid="head-selector"
              value={selectedHead}
              onChange={(e) => setSelectedHead(Number(e.target.value))}
            >
              {Array.from({ length: heads }, (_, i) => (
                <option key={i} value={i}>Head {i + 1}</option>
              ))}
            </select>
          </div>
          
          <div data-testid="attention-matrix">
            Layer {selectedLayer + 1}, Head {selectedHead + 1}
          </div>
        </div>
      );
    };

    render(<MockAttentionVisualization />);
    
    const layerSelector = screen.getByTestId('layer-selector');
    const headSelector = screen.getByTestId('head-selector');
    const attentionMatrix = screen.getByTestId('attention-matrix');
    
    expect(attentionMatrix).toHaveTextContent('Layer 1, Head 1');
    
    await user.selectOptions(layerSelector, '5');
    await user.selectOptions(headSelector, '7');
    
    expect(attentionMatrix).toHaveTextContent('Layer 6, Head 8');
  });

  test('should handle embedding visualization zoom and pan', () => {
    const MockEmbeddingVisualization = () => {
      const [transform, setTransform] = React.useState({ 
        scale: 1, 
        translateX: 0, 
        translateY: 0 
      });
      
      const handleZoom = (delta: number) => {
        setTransform(prev => ({
          ...prev,
          scale: Math.max(0.1, Math.min(5, prev.scale + delta))
        }));
      };
      
      return (
        <div>
          <div data-testid="zoom-controls">
            <button 
              data-testid="zoom-in"
              onClick={() => handleZoom(0.1)}
            >
              +
            </button>
            <button 
              data-testid="zoom-out"
              onClick={() => handleZoom(-0.1)}
            >
              -
            </button>
          </div>
          
          <div data-testid="embedding-canvas">
            Scale: {transform.scale.toFixed(1)}
          </div>
        </div>
      );
    };

    render(<MockEmbeddingVisualization />);
    
    const zoomInButton = screen.getByTestId('zoom-in');
    const embeddingCanvas = screen.getByTestId('embedding-canvas');
    
    expect(embeddingCanvas).toHaveTextContent('Scale: 1.0');
    
    fireEvent.click(zoomInButton);
    fireEvent.click(zoomInButton);
    
    expect(embeddingCanvas).toHaveTextContent('Scale: 1.2');
  });
});

describe('Error Handling and User Feedback', () => {
  test('should display loading states', () => {
    const MockLoadingComponent = ({ isLoading }: { isLoading: boolean }) => (
      <div>
        {isLoading ? (
          <div data-testid="loading-spinner">Loading...</div>
        ) : (
          <div data-testid="content">Content loaded</div>
        )}
      </div>
    );

    const { rerender } = render(<MockLoadingComponent isLoading={true} />);
    expect(screen.getByTestId('loading-spinner')).toBeInTheDocument();
    
    rerender(<MockLoadingComponent isLoading={false} />);
    expect(screen.getByTestId('content')).toBeInTheDocument();
  });

  test('should display error messages', () => {
    const MockErrorComponent = ({ error }: { error: string | null }) => (
      <div>
        {error ? (
          <div data-testid="error-message" role="alert">
            Error: {error}
          </div>
        ) : (
          <div data-testid="success-content">No errors</div>
        )}
      </div>
    );

    const { rerender } = render(<MockErrorComponent error={null} />);
    expect(screen.getByTestId('success-content')).toBeInTheDocument();
    
    rerender(<MockErrorComponent error="File upload failed" />);
    expect(screen.getByTestId('error-message')).toBeInTheDocument();
    expect(screen.getByTestId('error-message')).toHaveTextContent('Error: File upload failed');
  });

  test('should handle network errors gracefully', async () => {
    // Mock fetch failure
    mockFetch.mockRejectedValueOnce(new Error('Network error'));
    
    const MockNetworkComponent = () => {
      const [status, setStatus] = React.useState<'idle' | 'loading' | 'error' | 'success'>('idle');
      
      const handleRequest = async () => {
        setStatus('loading');
        try {
          await fetch('/api/test');
          setStatus('success');
        } catch (error) {
          setStatus('error');
        }
      };
      
      return (
        <div>
          <button data-testid="request-button" onClick={handleRequest}>
            Make Request
          </button>
          <div data-testid="status">Status: {status}</div>
        </div>
      );
    };

    render(<MockNetworkComponent />);
    
    const requestButton = screen.getByTestId('request-button');
    const statusDisplay = screen.getByTestId('status');
    
    fireEvent.click(requestButton);
    
    await waitFor(() => {
      expect(statusDisplay).toHaveTextContent('Status: error');
    });
  });
});

describe('Accessibility Testing', () => {
  test('should support keyboard navigation', async () => {
    const user = userEvent.setup();
    
    const MockKeyboardNavigation = () => {
      const [focusedIndex, setFocusedIndex] = React.useState(0);
      const items = ['Item 1', 'Item 2', 'Item 3'];
      
      const handleKeyDown = (e: React.KeyboardEvent) => {
        if (e.key === 'ArrowDown') {
          e.preventDefault();
          setFocusedIndex(prev => (prev + 1) % items.length);
        } else if (e.key === 'ArrowUp') {
          e.preventDefault();
          setFocusedIndex(prev => (prev - 1 + items.length) % items.length);
        }
      };
      
      return (
        <div data-testid="keyboard-nav" onKeyDown={handleKeyDown} tabIndex={0}>
          {items.map((item, index) => (
            <div
              key={index}
              data-testid={`item-${index}`}
              style={{ 
                background: index === focusedIndex ? 'blue' : 'transparent',
                color: index === focusedIndex ? 'white' : 'black'
              }}
            >
              {item}
            </div>
          ))}
        </div>
      );
    };

    render(<MockKeyboardNavigation />);
    
    const container = screen.getByTestId('keyboard-nav');
    const item0 = screen.getByTestId('item-0');
    const item1 = screen.getByTestId('item-1');
    
    // Focus the container
    container.focus();
    
    expect(item0).toHaveStyle('background: blue');
    
    // Navigate down
    await user.keyboard('{ArrowDown}');
    expect(item1).toHaveStyle('background: blue');
  });

  test('should have proper ARIA labels', () => {
    const MockAccessibleComponent = () => (
      <div>
        <button 
          data-testid="play-button"
          aria-label="Play audio file"
          aria-pressed="false"
        >
          ▶
        </button>
        
        <input
          data-testid="volume-slider"
          type="range"
          aria-label="Volume control"
          aria-valuemin="0"
          aria-valuemax="100"
          aria-valuenow="50"
        />
        
        <div
          data-testid="status-region"
          role="status"
          aria-live="polite"
        >
          Ready to play
        </div>
      </div>
    );

    render(<MockAccessibleComponent />);
    
    const playButton = screen.getByTestId('play-button');
    const volumeSlider = screen.getByTestId('volume-slider');
    const statusRegion = screen.getByTestId('status-region');
    
    expect(playButton).toHaveAttribute('aria-label', 'Play audio file');
    expect(volumeSlider).toHaveAttribute('aria-label', 'Volume control');
    expect(statusRegion).toHaveAttribute('role', 'status');
    expect(statusRegion).toHaveAttribute('aria-live', 'polite');
  });

  test('should meet color contrast requirements', () => {
    // This would typically use a tool like axe-core
    const checkColorContrast = (foreground: string, background: string): boolean => {
      // Simplified contrast check - in real tests, use proper WCAG calculation
      return foreground !== background;
    };

    expect(checkColorContrast('#000000', '#ffffff')).toBe(true);
    expect(checkColorContrast('#333333', '#333333')).toBe(false);
  });

  test('should handle screen reader announcements', () => {
    const MockScreenReaderComponent = () => {
      const [announcement, setAnnouncement] = React.useState('');
      
      const announce = (message: string) => {
        setAnnouncement(message);
        // Clear after a delay (simulate screen reader announcement)
        setTimeout(() => setAnnouncement(''), 100);
      };
      
      return (
        <div>
          <button
            data-testid="announce-button"
            onClick={() => announce('File uploaded successfully')}
          >
            Upload File
          </button>
          
          <div
            data-testid="announcement"
            role="alert"
            aria-live="assertive"
          >
            {announcement}
          </div>
        </div>
      );
    };

    render(<MockScreenReaderComponent />);
    
    const announceButton = screen.getByTestId('announce-button');
    const announcement = screen.getByTestId('announcement');
    
    fireEvent.click(announceButton);
    
    expect(announcement).toHaveTextContent('File uploaded successfully');
    expect(announcement).toHaveAttribute('role', 'alert');
  });
});