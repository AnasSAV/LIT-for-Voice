import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";

interface UploadedFile {
  file_id: string;
  filename: string;
  file_path: string;
  message: string;
  size?: number;
  duration?: number;
  sample_rate?: number;
}

interface Wav2Vec2Prediction {
  predicted_emotion: string;
  probabilities: Record<string, number>;
  confidence: number;
  ground_truth_emotion?: string;
}

interface WhisperPrediction {
  predicted_transcript: string;
  ground_truth: string;
  accuracy_percentage: number | null;
  word_error_rate: number | null;
  character_error_rate: number | null;
  levenshtein_distance: number | null;
  exact_match: number | null;
  character_similarity: number | null;
  word_count_predicted: number;
  word_count_truth: number;
}

interface PredictionDisplayProps {
  selectedFile?: UploadedFile | null;
  selectedEmbeddingFile?: string | null;
  model?: string;
  wav2vecPrediction?: Wav2Vec2Prediction | null;
  whisperPrediction?: WhisperPrediction | null;
  perturbedPredictions?: Wav2Vec2Prediction | WhisperPrediction | null;
  isLoading?: boolean;
  isLoadingPerturbed?: boolean;
  error?: string | null;
}

export const PredictionDisplay = ({
  selectedFile,
  selectedEmbeddingFile,
  model,
  wav2vecPrediction,
  whisperPrediction,
  perturbedPredictions,
  isLoading,
  isLoadingPerturbed,
  error
}: PredictionDisplayProps) => {
  if (!selectedFile && !selectedEmbeddingFile) {
    return (
      <Card className="border-gray-200 bg-white">
        <CardContent className="p-4 text-center text-gray-500">
          <div className="text-sm-tight">No file selected</div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className="border-gray-200 bg-white">
      <CardHeader className="pb-2">
        <CardTitle className="text-info">
          {model === "wav2vec2" ? "Classification Results" : model?.includes("whisper") ? "Transcription Results" : "Prediction Results"}
          {model === "wav2vec2" && (
            <Badge variant="secondary" className="ml-2 text-xs-tight">
              Wav2Vec2 Emotion
            </Badge>
          )}
          {model?.includes("whisper") && (
            <Badge variant="secondary" className="ml-2 text-xs-tight">
              {model.includes("large") ? "Whisper Large" : "Whisper Base"}
            </Badge>
          )}
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-2">
        {isLoading && (
          <div className="text-xs-tight text-gray-500 flex items-center gap-2">
            <div className="w-3 h-3 border-2 border-blue-500 border-t-transparent rounded-full animate-spin"></div>
            Loading prediction...
          </div>
        )}
        
        {error && (
          <div className="text-xs-tight text-red-500 p-2 bg-red-50 rounded border">
            Error: {error}
          </div>
        )}
        
        {model === "wav2vec2" && wav2vecPrediction && !isLoading ? (
          // Display wav2vec2 emotion predictions with comparison in two columns
          <div className="space-y-3">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {/* Original Prediction */}
              <div className="space-y-2 border-r md:border-r border-gray-200 pr-2">
                <div className="text-xs-tight font-medium flex items-center gap-2">
                  Original Audio
                  <span className="text-xs-tight text-gray-500 border border-gray-300 px-1 rounded">Original</span>
                </div>
                {Object.entries(wav2vecPrediction.probabilities)
                  .sort(([,a], [,b]) => b - a)
                  .map(([emotion, probability]) => {
                    const isPredicted = emotion === wav2vecPrediction.predicted_emotion;
                    return (
                      <div key={emotion} className="flex items-center justify-between text-xs-tight">
                        <div className="flex items-center gap-2">
                          <span className="capitalize">{emotion}</span>
                          {isPredicted && <span className="text-xs-tight text-gray-600 font-medium">Predicted</span>}
                        </div>
                        <div className="flex items-center gap-2 flex-1 max-w-[120px]">
                          <Progress value={probability * 100} className="h-2" />
                          <span className="text-muted-foreground min-w-[2rem]">
                            {(probability * 100).toFixed(1)}%
                          </span>
                        </div>
                      </div>
                    );
                  })}
              </div>

              {/* Perturbed Prediction */}
              <div className="space-y-2 pl-2">
                <div className="text-xs-tight font-medium flex items-center gap-2">
                  Perturbed Audio
                  <span className="text-xs-tight text-gray-500 border border-gray-300 px-1 rounded">Perturbed</span>
                  {isLoadingPerturbed && (
                    <div className="w-3 h-3 border-2 border-gray-400 border-t-transparent rounded-full animate-spin"></div>
                  )}
                </div>
                {!isLoadingPerturbed && perturbedPredictions && Object.entries((perturbedPredictions as Wav2Vec2Prediction).probabilities)
                  .sort(([,a], [,b]) => b - a)
                  .map(([emotion, probability]) => {
                    const isPredicted = emotion === (perturbedPredictions as Wav2Vec2Prediction).predicted_emotion;
                    const originalProb = wav2vecPrediction.probabilities[emotion] || 0;
                    const change = (probability - originalProb) * 100;
                    return (
                      <div key={emotion} className="flex items-center justify-between text-xs-tight">
                        <div className="flex items-center gap-2">
                          <span className="capitalize">{emotion}</span>
                          {isPredicted && <span className="text-xs-tight text-gray-600 font-medium">Predicted</span>}
                        </div>
                        <div className="flex items-center gap-2 flex-1 max-w-[120px]">
                          <Progress value={probability * 100} className="h-2" />
                          <span className="text-muted-foreground min-w-[2rem]">
                            {(probability * 100).toFixed(1)}%
                          </span>
                          <span className={`text-[10px] min-w-[2rem] ${
                            change > 0 ? "text-green-600" : change < 0 ? "text-red-600" : "text-muted-foreground"
                          }`}>
                            {change > 0 ? "+" : ""}{change.toFixed(1)}%
                          </span>
                        </div>
                      </div>
                    );
                  })}
              </div>
            </div>
          </div>
        ) : model?.includes("whisper") && whisperPrediction && !isLoading ? (
          // Improved UI for Transcription Results (metrics layout and clearer sections)
          <div className="space-y-3">
            <div className="grid grid-cols-1 gap-4">
              <div className="flex items-start justify-between gap-4">
                <div className="flex-1">
                  <div className="text-xs font-semibold">Transcription Metrics</div>
                  {whisperPrediction.accuracy_percentage !== null && whisperPrediction.word_error_rate !== null ? (
                    // Show metrics when ground truth is available
                    <div className="mt-2 grid grid-cols-2 sm:grid-cols-3 gap-2 text-xs">
                      <div className="p-2 bg-gray-50 rounded border text-gray-700">
                        <div className="text-[10px] text-gray-500">WER</div>
                        <div className="font-medium">{whisperPrediction.word_error_rate.toFixed(3)}</div>
                      </div>
                      <div className="p-2 bg-gray-50 rounded border text-gray-700">
                        <div className="text-[10px] text-gray-500">CER</div>
                        <div className="font-medium">{whisperPrediction.character_error_rate.toFixed(3)}</div>
                      </div>
                      <div className="p-2 bg-gray-50 rounded border text-gray-700">
                        <div className="text-[10px] text-gray-500">Accuracy</div>
                        <div className="font-medium">{whisperPrediction.accuracy_percentage.toFixed(1)}%</div>
                      </div>
                      <div className="p-2 bg-gray-50 rounded border text-gray-700">
                        <div className="text-[10px] text-gray-500">Words (Pred)</div>
                        <div className="font-medium">{whisperPrediction.word_count_predicted}</div>
                      </div>
                      <div className="p-2 bg-gray-50 rounded border text-gray-700">
                        <div className="text-[10px] text-gray-500">Words (Truth)</div>
                        <div className="font-medium">{whisperPrediction.word_count_truth}</div>
                      </div>
                      <div className="p-2 bg-gray-50 rounded border text-gray-700">
                        <div className="text-[10px] text-gray-500">Levenshtein</div>
                        <div className="font-medium">{whisperPrediction.levenshtein_distance}</div>
                      </div>
                    </div>
                  ) : (
                    // Show message when ground truth is not available
                    <div className="mt-2 p-3 bg-yellow-50 rounded border border-yellow-200 text-xs text-yellow-700">
                      <div className="font-medium">No Ground Truth Available</div>
                      <div className="mt-1">Accuracy metrics are not available for this dataset-model combination.</div>
                    </div>
                  )}
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <div className="text-xs font-medium">Predicted Transcript</div>
                  <div className="text-xs p-2 bg-blue-50 rounded border font-mono whitespace-pre-wrap">
                    {whisperPrediction.predicted_transcript ? `"${whisperPrediction.predicted_transcript}"` : <span className="italic text-gray-400">No prediction</span>}
                  </div>
                </div>

                {whisperPrediction.ground_truth && (
                  <div>
                    <div className="text-xs font-medium">Ground Truth</div>
                    <div className="text-xs p-2 bg-green-50 rounded border font-mono whitespace-pre-wrap">
                      {`"${whisperPrediction.ground_truth}"`}
                    </div>
                  </div>
                )}
              </div>

              {perturbedPredictions && (
                <div className="pt-2 border-t border-gray-100">
                  <div className="text-xs font-semibold">Perturbed Prediction</div>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-2">
                    <div>
                      {/* perturbed metrics if available */}
                      {typeof perturbedPredictions === 'object' && (
                        <div className="grid grid-cols-2 gap-2 text-xs text-gray-600">
                          <div>WER: {(perturbedPredictions as WhisperPrediction).word_error_rate?.toFixed(3) || 'N/A'}</div>
                          <div>CER: {(perturbedPredictions as WhisperPrediction).character_error_rate?.toFixed(3) || 'N/A'}</div>
                          <div>Words P: {(perturbedPredictions as WhisperPrediction).word_count_predicted || 'N/A'}</div>
                          <div>Words T: {(perturbedPredictions as WhisperPrediction).word_count_truth || 'N/A'}</div>
                          <div>Accuracy: {(perturbedPredictions as WhisperPrediction).accuracy_percentage?.toFixed(1) || 'N/A'}%</div>
                          <div>Levenshtein: {(perturbedPredictions as WhisperPrediction).levenshtein_distance || 'N/A'}</div>
                        </div>
                      )}
                    </div>

                    <div>
                      <div className="text-xs font-medium">Predicted Transcript (Perturbed)</div>
                      <div className="mt-1 text-xs p-2 bg-white rounded border font-mono whitespace-pre-wrap text-gray-800">
                        {(perturbedPredictions as WhisperPrediction).predicted_transcript ? `"${(perturbedPredictions as WhisperPrediction).predicted_transcript}"` : <span className="italic text-gray-400">No prediction</span>}
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        ) : !model?.includes("whisper") && model !== "wav2vec2" ? (
          // Display placeholder/mock data for other models
          [
            { label: "Neutral", probability: 0.87, isTrue: true, isPredicted: true },
            { label: "Happy", probability: 0.08, isTrue: false, isPredicted: false },
            { label: "Sad", probability: 0.03, isTrue: false, isPredicted: false },
            { label: "Angry", probability: 0.02, isTrue: false, isPredicted: false },
          ].map((item, idx) => (
            <div key={idx} className="flex items-center justify-between text-xs">
              <div className="flex items-center gap-2">
                <span>{item.label}</span>
                {item.isPredicted && <Badge variant="default" className="text-[10px] px-1">P</Badge>}
                {item.isTrue && <Badge variant="outline" className="text-[10px] px-1">T</Badge>}
              </div>
              <div className="flex items-center gap-2 flex-1 max-w-[120px]">
                <Progress value={item.probability * 100} className="h-2" />
                <span className="text-muted-foreground min-w-[2rem]">
                  {(item.probability * 100).toFixed(0)}%
                </span>
              </div>
            </div>
          ))
        ) : null}
      </CardContent>
    </Card>
  );
};