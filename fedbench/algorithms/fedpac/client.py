import torch
import traceback
from torch import nn
from torch.utils.data import DataLoader
from flwr.client import NumPyClient
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters


class FedPACClient(NumPyClient):
    def __init__(self, client_id, train_dataset, test_dataset, model, args):
        self.cid = client_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.lambda_reg = args.lambda_reg
        self.global_centroids = None
        self.args = args

        self.train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        if not (hasattr(self.model, "feature_extractor") and hasattr(self.model, "classifier")):
            raise ValueError(f"[Client {self.cid}] Model must have 'feature_extractor' and 'classifier'")

    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        try:
            state_dict = dict(zip(self.model.state_dict().keys(), [torch.tensor(p) for p in parameters]))
            self.model.load_state_dict(state_dict, strict=True)
        except Exception as e:
            print(f"[Client {self.cid}] ERROR in set_parameters: {e}")
            raise

    def fit(self, parameters, config):
        try:
            self.set_parameters(parameters)

            # Update round if passed in config
            self.args.round = int(config.get("server_round", 0))

            # Load global centroids
            if "global_centroids" in config:
                self.global_centroids = {
                    int(k): torch.tensor(v).to(self.device)
                    for k, v in config["global_centroids"].items()
                }

            updated_weights, local_stats = self.train()

            # Calculate training metrics
            train_loss, train_acc = self._evaluate_on_trainset()

            # Prepare metrics dictionary with proper serialization
            metrics = {
                "loss": float(train_loss),
                "accuracy": float(train_acc)
            }
            
            # Serialize centroids and sample counts as separate flat entries
            if "centroids" in local_stats:
                for class_id, centroid in local_stats["centroids"].items():
                    # Convert centroid to list and store as string representation
                    centroid_key = f"centroid_{class_id}"
                    metrics[centroid_key] = str(centroid.cpu().numpy().tolist())
                    
            if "sample_counts" in local_stats:
                for class_id, count in local_stats["sample_counts"].items():
                    count_key = f"sample_count_{class_id}"
                    metrics[count_key] = int(count)

            # Return the correct format: (parameters, num_examples, metrics)
            return (
                self.get_parameters({}),  # Return parameters as list of numpy arrays
                len(self.train_loader.dataset),  # Number of examples
                metrics  # Metrics dictionary with flat structure
            )

        except Exception as e:
            print(f"[Client {self.cid}] CRITICAL ERROR in fit(): {e}")
            print(traceback.format_exc())
            # Return valid default values in case of error
            return self.get_parameters({}), 0, {"loss": 0.0, "accuracy": 0.0}

    def train(self):
        self.model.train()

        # Phase 1: Train classifier with frozen feature extractor
        self.model.feature_extractor.eval()
        self.model.classifier.train()
        
        # Freeze feature extractor parameters
        for param in self.model.feature_extractor.parameters():
            param.requires_grad = False
        
        optimizer_g = torch.optim.SGD(self.model.classifier.parameters(), lr=self.args.lr_g, momentum=0.9)
        
        for epoch in range(self.args.epochs // 2):
            for x, y in self.train_loader:
                x, y = x.to(self.device), y.to(self.device)
                
                # Feature extractor in eval mode, no need for no_grad
                features = self.model.feature_extractor(x)
                outputs = self.model.classifier(features)
                loss = self.criterion(outputs, y)
                
                optimizer_g.zero_grad()
                loss.backward()
                optimizer_g.step()

        # Phase 2: Train feature extractor with frozen classifier
        self.model.feature_extractor.train()
        self.model.classifier.train()  # Keep in train mode but freeze parameters
        
        # Unfreeze feature extractor, freeze classifier
        for param in self.model.feature_extractor.parameters():
            param.requires_grad = True
        for param in self.model.classifier.parameters():
            param.requires_grad = False
        
        optimizer_f = torch.optim.SGD(self.model.feature_extractor.parameters(), lr=self.args.lr_f, momentum=0.9)
        
        for epoch in range(self.args.epochs // 2):
            total_loss, correct, total = 0.0, 0, 0
            
            for x, y in self.train_loader:
                x, y = x.to(self.device), y.to(self.device)
                
                # Both parts need to be in the computation graph
                features = self.model.feature_extractor(x)
                outputs = self.model.classifier(features)
                
                loss = self.criterion(outputs, y)
                
                # Feature alignment regularization
                if self.global_centroids and self.args.round >= self.args.centroid_start_round:
                    reg_loss = self._feature_alignment_loss(features, y)
                    loss = loss + self.lambda_reg * reg_loss
                
                optimizer_f.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.feature_extractor.parameters(), max_norm=10.0)
                optimizer_f.step()
                
                total_loss += loss.item() * y.size(0)
                correct += (outputs.argmax(1) == y).sum().item()
                total += y.size(0)
            
            acc = correct / total if total > 0 else 0.0
            avg_loss = total_loss / total if total > 0 else 0.0
            print(f"[Client {self.cid}] Epoch {epoch+1} - Loss: {avg_loss:.4f}, Acc: {acc:.4f}")

        # Unfreeze all parameters for next round
        for param in self.model.parameters():
            param.requires_grad = True
            
        # Set both parts to eval mode before computing centroids
        self.model.eval()
        
        return self.model.state_dict(), self._compute_local_stats()

    def evaluate(self, parameters, config):
        try:
            self.set_parameters(parameters)
            self.model.eval()

            total_loss, correct, total = 0.0, 0, 0
            with torch.no_grad():
                for x, y in self.test_loader:
                    x, y = x.to(self.device), y.to(self.device)
                    features = self.model.feature_extractor(x)
                    output = self.model.classifier(features)
                    loss = self.criterion(output, y)
                    total_loss += loss.item() * y.size(0)
                    correct += (output.argmax(1) == y).sum().item()
                    total += y.size(0)

            loss = total_loss / total if total > 0 else float("inf")
            accuracy = correct / total if total > 0 else 0.0

            return loss, total, {"accuracy": accuracy}

        except Exception as e:
            print(f"[Client {self.cid}] ERROR in evaluate(): {e}")
            print(traceback.format_exc())
            return float("inf"), 1, {"accuracy": 0.0}

    def _evaluate_on_trainset(self):
        """Evaluate on training set for metrics reporting"""
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0
        
        with torch.no_grad():
            for x, y in self.train_loader:
                x, y = x.to(self.device), y.to(self.device)
                features = self.model.feature_extractor(x)
                output = self.model.classifier(features)
                loss = self.criterion(output, y)
                total_loss += loss.item() * y.size(0)
                correct += (output.argmax(1) == y).sum().item()
                total += y.size(0)
        
        return total_loss / total if total > 0 else 0.0, correct / total if total > 0 else 0.0

    def _feature_alignment_loss(self, features, labels):
        """Compute feature alignment loss with global centroids"""
        reg_loss = 0.0
        count = 0
        
        for i in range(features.size(0)):
            label = labels[i].item()
            if label in self.global_centroids:
                centroid = self.global_centroids[label]
                # L2 distance between feature and centroid
                reg_loss += torch.norm(features[i] - centroid, p=2) ** 2
                count += 1
        
        return reg_loss / count if count > 0 else 0.0

    def _compute_local_stats(self):
        """Compute local centroids in feature space"""
        self.model.eval()
        class_features = {}
        class_counts = {}

        with torch.no_grad():
            for x, y in self.train_loader:
                x, y = x.to(self.device), y.to(self.device)
                features = self.model.feature_extractor(x)
                
                for i in range(x.size(0)):
                    label = y[i].item()
                    if label not in class_features:
                        class_features[label] = torch.zeros_like(features[i])
                        class_counts[label] = 0
                    
                    class_features[label] += features[i]
                    class_counts[label] += 1

        # Compute average features (centroids) for each class
        centroids = {
            k: class_features[k] / class_counts[k]
            for k in class_features if class_counts[k] > 0
        }
        
        return {"centroids": centroids, "sample_counts": class_counts}