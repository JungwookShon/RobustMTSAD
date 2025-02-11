import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
from utils.utils import *
from model.AnomalyTransformer import AnomalyTransformer
from data_factory.data_loader import get_loader_segment
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pandas as pd

from pot import pot
from sklearn.metrics import precision_recall_fscore_support, accuracy_score



def my_kl_loss(p, q):
    res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
    return torch.mean(torch.sum(res, dim=-1), dim=1)


def adjust_learning_rate(optimizer, epoch, lr_):
    lr_adjust = {epoch: lr_ * (0.5 ** ((epoch - 1) // 1))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, dataset_name='', delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_score2 = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.val_loss2_min = np.Inf
        self.delta = delta
        self.dataset = dataset_name

    def __call__(self, val_loss, val_loss2, model, path):
        score = -val_loss
        score2 = -val_loss2
        print(f"score: {score}, score2: {score2}")
        if self.best_score is None:
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(val_loss, val_loss2, model, path)
        elif score < self.best_score + self.delta or score2 > self.best_score2 + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(val_loss, val_loss2, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, val_loss2, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), os.path.join(path, str(self.dataset) + '_checkpoint.pth'))
        self.val_loss_min = val_loss
        self.val_loss2_min = val_loss2


class Solver(object):
    DEFAULTS = {}

    def __init__(self, config):

        self.__dict__.update(Solver.DEFAULTS, **config)

        self.train_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                               mode='train',
                                               dataset=self.dataset,
                                               group=self.group)
        self.vali_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                              mode='val',
                                              dataset=self.dataset,
                                              group=self.group)
        self.test_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                              mode='test',
                                              dataset=self.dataset,
                                              group=self.group) # add group
        self.thre_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                              mode='thre',
                                              dataset=self.dataset,
                                              group=self.group)

        self.build_model()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.MSELoss()

    def build_model(self):
        self.model = AnomalyTransformer(win_size=self.win_size, enc_in=self.input_c, c_out=self.output_c, e_layers=3)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        if torch.cuda.is_available():
            self.model.cuda()

    def adjust_detection(self, pred, gt):
        """
        Adjusts the predicted anomalies (pred) based on ground truth (gt).
        
        Parameters:
            pred (list or np.ndarray): The predicted anomalies (binary: 0 or 1).
            gt (list or np.ndarray): The ground truth anomalies (binary: 0 or 1).

        Returns:
            np.ndarray: The adjusted predictions.
        """
        # Ensure inputs are lists for easy manipulation
        if isinstance(pred, np.ndarray):
            pred = pred.tolist()
        if isinstance(gt, np.ndarray):
            gt = gt.tolist()

        anomaly_state = False

        for i in range(len(gt)):
            if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
                anomaly_state = True
                # Adjust anomalies backwards
                for j in range(i, 0, -1):
                    if gt[j] == 0:
                        break
                    else:
                        if pred[j] == 0:
                            pred[j] = 1
                # Adjust anomalies forwards
                for j in range(i, len(gt)):
                    if gt[j] == 0:
                        break
                    else:
                        if pred[j] == 0:
                            pred[j] = 1
            elif gt[i] == 0:
                anomaly_state = False

            if anomaly_state:
                pred[i] = 1

        # Convert back to numpy array for consistency
        pred = np.array(pred)
        return pred

    def vali(self, vali_loader):
        self.model.eval()

        loss_1 = []
        loss_2 = []
        for i, (input_data, _) in enumerate(vali_loader):
            input = input_data.float().to(self.device)
            output, series, prior, _ = self.model(input)
            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                series_loss += (torch.mean(my_kl_loss(series[u], (
                        prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                               self.win_size)).detach())) + torch.mean(
                    my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)).detach(),
                        series[u])))
                prior_loss += (torch.mean(
                    my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                       self.win_size)),
                               series[u].detach())) + torch.mean(
                    my_kl_loss(series[u].detach(),
                               (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                       self.win_size)))))
            series_loss = series_loss / len(prior)
            prior_loss = prior_loss / len(prior)

            rec_loss = self.criterion(output, input)
            loss_1.append((rec_loss - self.k * series_loss).item())
            loss_2.append((rec_loss + self.k * prior_loss).item())

        return np.average(loss_1), np.average(loss_2)

    def train(self):

        print("======================TRAIN MODE======================")

        time_now = time.time()
        path = self.model_save_path
        if not os.path.exists(path):
            os.makedirs(path)
        early_stopping = EarlyStopping(patience=3, verbose=True, dataset_name=self.dataset, delta=0.1)
        
        train_steps = len(self.train_loader)


        # Group List
        # train_groups = ['1-1','2-1','4-2','4-3','6-1','7-1'] # SEC dataset
        train_groups = ['1-1','1-2','1-3','1-4','1-5','1-6','1-7','1-8',
                         '2-1','2-2','2-3','2-4','2-5','2-6','2-7','2-8','2-9',
                         '3-1','3-2','3-3','3-4','3-5','3-6','3-7','3-8','3-9','3-10','3-11'] # SMD dataset

        for epoch in range(self.num_epochs):
            iter_count = 0
            loss1_list = []

            epoch_time = time.time()
            self.model.train()

            for group_name in train_groups:
                # print(f"Train Group: {group_name}")

                self.train_loader = get_loader_segment(data_path=self.data_path, 
                                                    batch_size=self.batch_size, 
                                                    win_size=self.win_size,
                                                    mode='train',
                                                    dataset=self.dataset,
                                                    group=group_name)


                for i, (input_data, labels) in enumerate(self.train_loader):

                    self.optimizer.zero_grad()
                    iter_count += 1
                    input = input_data.float().to(self.device)

                    output, series, prior, _ = self.model(input)

                    # calculate Association discrepancy
                    series_loss = 0.0
                    prior_loss = 0.0
                    for u in range(len(prior)):
                        series_loss += (torch.mean(my_kl_loss(series[u], (
                                prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                    self.win_size)).detach())) + torch.mean(
                            my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                            self.win_size)).detach(),
                                    series[u])))
                        prior_loss += (torch.mean(my_kl_loss(
                            (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                    self.win_size)),
                            series[u].detach())) + torch.mean(
                            my_kl_loss(series[u].detach(), (
                                    prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                        self.win_size)))))
                    series_loss = series_loss / len(prior)
                    prior_loss = prior_loss / len(prior)

                    rec_loss = self.criterion(output, input) 

                    loss1 = rec_loss - self.k * series_loss
                    loss2 = rec_loss + self.k * prior_loss
                    loss1_list.append(loss1.item())

                    if (i + 1) % 100 == 0:
                        speed = (time.time() - time_now) / iter_count
                        left_time = speed * ((self.num_epochs - epoch) * train_steps - i)
                        print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                        iter_count = 0
                        time_now = time.time()

                    # Minimax strategy
                    loss1.backward(retain_graph=True)
                    loss2.backward()
                    self.optimizer.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(loss1_list)

            vali_loss1_list = []
            vali_loss2_list = []

            for group_name in train_groups:
                self.vali_loader = get_loader_segment(self.data_path, 
                                                batch_size=self.batch_size, 
                                                win_size=self.win_size,
                                                mode='val',
                                                dataset=self.dataset,
                                                group=group_name)

                vali_loss1, vali_loss2 = self.vali(self.vali_loader)
                vali_loss1_list.append(vali_loss1)
                vali_loss2_list.append(vali_loss2)

            vali_loss1 = sum(vali_loss1_list) / len(vali_loss1_list)
            vali_loss2 = sum(vali_loss2_list) / len(vali_loss2_list)


            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} ".format(
                    epoch + 1, train_steps, train_loss, vali_loss1))
            early_stopping(vali_loss1, vali_loss2, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(self.optimizer, epoch + 1, self.lr)



    # implement pot algorithm start

    def test_pot(self, group_name, max_energy):
        # Load the trained model
        self.model.load_state_dict(torch.load(os.path.join(str(self.model_save_path), str(self.dataset) + '_checkpoint.pth')))
        self.model.eval()
        print(f"Group Name: {group_name}")
        # Criterion for anomaly score calculation
        criterion = nn.MSELoss(reduction='none')

        test_error = []
        test_labels = []
        feature_losses = []  
        metric_based_losses = []

        temperature = 5

        self.test_loader = get_loader_segment(data_path=self.data_path, 
                                                batch_size=self.batch_size, 
                                                win_size=self.win_size,
                                                mode='test',
                                                dataset=self.dataset,
                                                group=group_name)

        for i, (input_data, labels) in enumerate(self.test_loader):
            input_data = input_data.float().to(self.device)  
            output, series, prior, _ = self.model(input_data)

            labels = labels.numpy()  
            
            loss = criterion(input_data, output)  
            feature_losses.append(loss.detach().cpu().numpy())  
            
            loss = torch.mean(loss, dim=-1)  
            test_error.append(loss.detach().cpu().numpy())
            test_labels.append(labels)  

            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                if u == 0:
                    series_loss = my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                            self.win_size)).detach()) * temperature
                    prior_loss = my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                            self.win_size)),series[u].detach()) * temperature
                else:
                    series_loss += my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                            self.win_size)).detach()) * temperature
                    prior_loss += my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                        self.win_size)),series[u].detach()) * temperature
            metric = torch.softmax((-series_loss - prior_loss), dim=-1)
            metric_based_loss = metric * loss
            metric_based_losses.append(metric_based_loss.detach().cpu().numpy())

        
        test_error = np.concatenate(test_error, axis=0).reshape(-1)  
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        feature_losses = np.concatenate(feature_losses, axis=0).reshape(-1, self.input_c)

        metric_based_losses = np.concatenate(metric_based_losses, axis=0).reshape(-1)
        # print(f"metric_based_losses.shape: {metric_based_losses.shape}")

        gt = test_labels.astype(int)
        

        # print("Applying POT...")
        # pot_threshold, _ = pot(test_error, risk=0.005)  # risk 
        # print("Applying POT for metric-based loss...")
        threshold_metric_loss, _ = pot(metric_based_losses, risk=0.0195) # control the risk
        # SMD risk = 0.007


        print(f"Thresholds:")
        # # print(f"  Test Error Threshold: {pot_threshold}")
        print(f"  Metric-based Loss Threshold: {threshold_metric_loss}")


        # max_energy = threshold_metric_loss
        if threshold_metric_loss < max_energy:
            threshold_metric_loss = max_energy
            print("Threshold Metric Loss is too low. Set to max energy.")
            print(f"  Metric-based Loss Threshold: {threshold_metric_loss}")
        
        # threshold_metric_loss = 0.46496197119354876

        # pred = (test_error > pot_threshold).astype(int)  

        pred_metric_loss = (metric_based_losses > threshold_metric_loss).astype(int)

        # print(f"\nAnomaly Predictions:")
        # # print(f"  Test Error Predictions: {np.sum(pred)} anomalies detected.")
        # print(f"  Metric-based Predictions: {np.sum(pred_metric_loss)} anomalies detected.")

        # output_file = f"{self.model_save_path}/{self.dataset}_{group_name}_AnomalyDetectionComparison.txt"
        # with open(output_file, "w") as f:
        #     f.write("Thresholds:\n")
        #     # f.write(f"  Test Error Threshold: {pot_threshold}\n")
        #     f.write(f"  Metric-based Loss Threshold: {threshold_metric_loss}\n\n")
        #     f.write("Anomaly Predictions:\n")
        #     # f.write(f"  Test Error Predictions: {np.sum(pred)} anomalies detected.\n")
        #     f.write(f"  Metric-based Predictions: {np.sum(pred_metric_loss)} anomalies detected.\n")

        # print(f"Results saved to {output_file}")
        

        # pred = self.adjust_detection(pred, gt)
        pred_metric_loss = self.adjust_detection(pred_metric_loss, gt)
        
        
        # accuracy = accuracy_score(gt, pred)
        # precision, recall, f_score, _ = precision_recall_fscore_support(gt, pred, average='binary')
        
        # print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f_score:.4f}")
        
        accuracy_metric_loss = accuracy_score(gt, pred_metric_loss)
        precision_metric_loss, recall_metric_loss, f_score_metric_loss, _ = precision_recall_fscore_support(gt, pred_metric_loss, average='binary')

        # print("Metric-based Loss:")
        print(f"Accuracy: {accuracy_metric_loss:.4f}, Precision: {precision_metric_loss:.4f}, Recall: {recall_metric_loss:.4f}, F1-score: {f_score_metric_loss:.4f}")



        # Create figure
        # fig = go.Figure()

        # # Plot test_error (as a line plot)
        # fig.add_trace(go.Scatter(
        #     x=list(range(len(test_error))), 
        #     y=test_error, 
        #     mode='lines', 
        #     name='test_error', 
        #     line=dict(color='#32CD32', width=1.5),  # Brighter green color
        #     opacity=0.8  # Slightly more opaque
        # ))

        # # Plot threshold line (as a horizontal line)
        # fig.add_trace(go.Scatter(
        #     x=[0, len(test_error)], 
        #     y=[pot_threshold, pot_threshold], 
        #     mode='lines', 
        #     name='Threshold', 
        #     line=dict(color='red', width=2, dash='dash')  # Red dashed line for better visibility
        # ))

        # # Fill area below the threshold
        # fig.add_trace(go.Scatter(
        #     x=list(range(len(test_error))), 
        #     y=[pot_threshold] * len(test_error), 
        #     mode='lines', 
        #     name='Threshold Fill', 
        #     fill='tonexty', 
        #     line=dict(color='rgba(255, 0, 0, 0)'),  # Transparent line
        #     fillcolor='rgba(255, 0, 0, 0.2)',  # Light red for filled area
        #     opacity=0.4, 
        #     showlegend=False
        # ))

        # # Update layout: add titles, background color, etc.
        # fig.update_layout(
        #     title="POT - Anomaly Detection",
        #     xaxis_title="Index",
        #     yaxis_title="Value",
        #     plot_bgcolor="#1c1c1c",  # Slightly lighter background for better contrast
        #     paper_bgcolor="#1c1c1c",
        #     font=dict(color='white'),  # Font color
        #     showlegend=True
        # )

        # # Customize axis lines and ticks
        # fig.update_xaxes(
        #     showline=True, 
        #     linewidth=1.5, 
        #     linecolor='white', 
        #     tickfont=dict(size=12, color='white')
        # )
        # fig.update_yaxes(
        #     showline=True, 
        #     linewidth=1.5, 
        #     linecolor='white', 
        #     tickfont=dict(size=12, color='white')#, 
        #     # type="log"
        # )

        # # Save figure to HTML file
        # fig.write_html(f"{self.model_save_path}/{self.dataset}_{group_name}_POT_AnomalyDetection.html")

        # Create figure for metric-based losses
        fig_metric = go.Figure()

        # Plot metric_based_losses (as a line plot)
        fig_metric.add_trace(go.Scatter(
            x=list(range(len(metric_based_losses))), 
            y=metric_based_losses, 
            mode='lines', 
            name='Anomaly Score', 
            line=dict(color='black', width=1.5),  
            opacity=0.8  # Slightly more opaque
        ))

        # Plot threshold line (as a horizontal line)
        fig_metric.add_trace(go.Scatter(
            x=[0, len(metric_based_losses)], 
            y=[threshold_metric_loss, threshold_metric_loss], 
            mode='lines', 
            name='Threshold', 
            line=dict(color='red', width=2, dash='dash')  # Red dashed line for better visibility
        ))

        # Update layout: add titles, background color, etc.
        fig_metric.update_layout(
            # title="Metric-based Losses - Anomaly Detection",
            title="Anomaly Detection with POT",
            xaxis_title="Time",
            yaxis_title="Anomaly Score",
            plot_bgcolor="white",  
            paper_bgcolor="white",
            font=dict(color='black'),  # Font color
            showlegend=True,
            legend=dict(
            x=0.8,  # Position legend inside the plot (x coordinate)
            y=0.9,  # Position legend inside the plot (y coordinate)
            bgcolor="rgba(255,255,255,0.7)",  # Semi-transparent white background
            bordercolor="black",
            borderwidth=1
            )
        )

        # Customize axis lines and ticks
        fig_metric.update_xaxes(
            showline=True, 
            linewidth=1.5, 
            linecolor='black', 
            tickfont=dict(size=12, color='black')
        )
        fig_metric.update_yaxes(
            showline=True, 
            linewidth=1.5, 
            linecolor='black', 
            tickfont=dict(size=12, color='black')#, 
            # type="log"
        )

        # Save figure to HTML file
        fig_metric.write_html(f"{self.model_save_path}/{self.dataset}_{group_name}_MetricBasedLoss_AnomalyDetection.html")





        # Find indices where ground truth is 1 and predictions are correct/incorrect
        correct_positive_indices = [i for i in range(len(pred_metric_loss)) if gt[i] == 1 and pred_metric_loss[i] == 1]
        incorrect_positive_indices = [i for i in range(len(pred_metric_loss)) if gt[i] == 1 and pred_metric_loss[i] != 1]
        overcorrect_positive_indices = [i for i in range(len(pred_metric_loss)) if gt[i] == 0 and pred_metric_loss[i] == 1]
        correct_predictions = len(correct_positive_indices)
        incorrect_predictions = len(incorrect_positive_indices)
        overcorrect_predictions = len(overcorrect_positive_indices)
        # Output the results
        print(f"Correct Positive Predictions: {correct_predictions}")
        print(f"Incorrect Positive Predictions: {incorrect_predictions}")
        print(f"Overcorrect Positive Predictions: {overcorrect_predictions}\n")

        # Save to file
        output_filename = f"{self.model_save_path}/{self.dataset}_{group_name}_positive_prediction_results.txt"
        with open(output_filename, "w") as f:
            f.write(f"Accuracy: {accuracy_metric_loss:.4f}, Precision: {precision_metric_loss:.4f}, Recall: {recall_metric_loss:.4f}, F1-score: {f_score_metric_loss:.4f}\n")
            f.write(f"Correct Positive Predictions: {correct_predictions}\n")
            f.write(f"Incorrect Positive Predictions: {incorrect_predictions}\n")
            f.write(f"Overcorrect Positive Predictions: {overcorrect_predictions}\n")
            f.write("Correct Positive Predictions (Ground Truth = 1 and Prediction = 1):\n")
            for i in correct_positive_indices:
                f.write(f"Index {i}: Prediction {pred_metric_loss[i]} -> Ground Truth {gt[i]}\n")
            f.write("\nIncorrect Positive Predictions (Ground Truth = 1 and Prediction != 1):\n")
            for i in incorrect_positive_indices:
                f.write(f"Index {i}: Prediction {pred_metric_loss[i]} -> Ground Truth {gt[i]}\n")
            f.write("\nOvercorrect Positive Predictions (Ground Truth = 0 and Prediction = 1):\n")
            for i in overcorrect_positive_indices:
                f.write(f"Index {i}: Prediction {pred_metric_loss[i]} -> Ground Truth {gt[i]}\n")   
            f.write("\nFeature losses:[index, mean_loss, feature_losses]\n")
            for index in np.where(pred_metric_loss == 1)[0]:  
                mean_loss = test_error[index]  
                feature_loss_str = ", ".join([f"{v:.3f}" for v in feature_losses[index]])  
                f.write(f"{index}, {mean_loss:.3f}, {feature_loss_str}\n")  
        
            

        return pred_metric_loss, gt, metric_based_losses, threshold_metric_loss

  


    # implement pot algorithm end

    def test(self):
        self.model.load_state_dict(
            torch.load(
                os.path.join(str(self.model_save_path), str(self.dataset) + '_checkpoint.pth')))
        self.model.eval()
        temperature = 5

        print("======================TEST MODE======================")

        criterion = nn.MSELoss(reduction='none')
        attens_energy = []

        # Group List
        # train_groups = ['1-1','2-1','4-2','4-3','6-1','7-1'] # SEC dataset 
        train_groups = ['1-1','1-2','1-3','1-4','1-5','1-6','1-7','1-8',
                         '2-1','2-2','2-3','2-4','2-5','2-6','2-7','2-8','2-9',
                         '3-1','3-2','3-3','3-4','3-5','3-6','3-7','3-8','3-9','3-10','3-11'] # SMD dataset

        
        for group_name in train_groups:
            # print(f"Attention Group: {group_name}")

            self.train_loader = get_loader_segment(data_path=self.data_path, 
                                                batch_size=self.batch_size, 
                                                win_size=self.win_size,
                                                mode='train',
                                                dataset=self.dataset,
                                                group=group_name)
        
        # (1) stastic on the train set
            
            for i, (input_data, labels) in enumerate(self.train_loader):
                input = input_data.float().to(self.device)
                output, series, prior, _ = self.model(input)
                loss = torch.mean(criterion(input, output), dim=-1)
                series_loss = 0.0
                prior_loss = 0.0

                for u in range(len(prior)):
                    if u == 0:
                        series_loss = my_kl_loss(series[u], (
                                prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                    self.win_size)).detach()) * temperature
                        prior_loss = my_kl_loss(
                            (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                    self.win_size)),
                            series[u].detach()) * temperature
                    else:
                        series_loss += my_kl_loss(series[u], (
                                prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                    self.win_size)).detach()) * temperature
                        prior_loss += my_kl_loss(
                            (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                    self.win_size)),
                            series[u].detach()) * temperature

                metric = torch.softmax((-series_loss - prior_loss), dim=-1)
                cri = metric * loss
                cri = cri.detach().cpu().numpy()
                attens_energy.append(cri)


        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        train_energy = np.array(attens_energy)
        # max_energy = np.max(train_energy)
        # train_energy_series = pd.Series(train_energy)
        # print(f"max of train_energy: {np.max(max_energy)}")
        # print(train_energy_series.describe())

        max_energy, _ = pot(train_energy, risk=0.015) # control the risk
        print("Threshold_train_POT :", max_energy)

        # (2) find the threshold
        attens_energy = []

        # test_groups = ['3-1','3-2','3-3','3-4','4-1','4-4','5-1','5-2'] # SEC dataset
        test_groups = ['1-1','1-2','1-3','1-4','1-5','1-6','1-7','1-8',
                         '2-1','2-2','2-3','2-4','2-5','2-6','2-7','2-8','2-9',
                         '3-1','3-2','3-3','3-4','3-5','3-6','3-7','3-8','3-9','3-10','3-11'] # SMD dataset
        for group_name in test_groups:
            # print(f"Threshold Group: {group_name}")

            self.thre_loader = get_loader_segment(data_path=self.data_path, 
                                                batch_size=self.batch_size, 
                                                win_size=self.win_size,
                                                mode='thre',
                                                dataset=self.dataset,
                                                group=group_name)

            for i, (input_data, labels) in enumerate(self.thre_loader):
                input = input_data.float().to(self.device)
                output, series, prior, _ = self.model(input)

                loss = torch.mean(criterion(input, output), dim=-1)

                series_loss = 0.0
                prior_loss = 0.0
                for u in range(len(prior)):
                    if u == 0:
                        series_loss = my_kl_loss(series[u], (
                                prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                    self.win_size)).detach()) * temperature
                        prior_loss = my_kl_loss(
                            (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                    self.win_size)),
                            series[u].detach()) * temperature
                    else:
                        series_loss += my_kl_loss(series[u], (
                                prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                    self.win_size)).detach()) * temperature
                        prior_loss += my_kl_loss(
                            (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                    self.win_size)),
                            series[u].detach()) * temperature
                # Metric
                metric = torch.softmax((-series_loss - prior_loss), dim=-1)
                cri = metric * loss
                cri = cri.detach().cpu().numpy()
                attens_energy.append(cri)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        combined_energy = np.concatenate([train_energy, test_energy], axis=0)
        thresh = np.percentile(combined_energy, 100 - self.anormly_ratio)
        print("Threshold :", thresh)


        # (3) evaluation on the test set
        test_labels = []
        attens_energy = []
        
        original_sequences = []
        predicted_sequences = []

        original_sequences2 = []
        predicted_sequences2 = []

        for group_name in test_groups:
            # print(f"Test Group: {group_name}")

            self.test_loader = get_loader_segment(data_path=self.data_path, 
                                                batch_size=self.batch_size, 
                                                win_size=self.win_size,
                                                mode='test',
                                                dataset=self.dataset,
                                                group=group_name)

            for i, (input_data, labels) in enumerate(self.test_loader):
                input = input_data.float().to(self.device)
                output, series, prior, _ = self.model(input)

                loss = torch.mean(criterion(input, output), dim=-1)

                series_loss = 0.0
                prior_loss = 0.0

                original_first_sequence = input_data[:, :, 1].cpu().numpy()  
                predicted_first_sequence = output[:, :, 1].detach().cpu().numpy()
                original_sequences.append(original_first_sequence)
                predicted_sequences.append(predicted_first_sequence)

                original_second_sequence = input_data[:, :, 2].cpu().numpy()  
                predicted_second_sequence = output[:, :, 2].detach().cpu().numpy()
                original_sequences2.append(original_second_sequence)
                predicted_sequences2.append(predicted_second_sequence)


                for u in range(len(prior)):
                    if u == 0:
                        series_loss = my_kl_loss(series[u], (
                                prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                    self.win_size)).detach()) * temperature
                        prior_loss = my_kl_loss(
                            (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                    self.win_size)),
                            series[u].detach()) * temperature
                    else:
                        series_loss += my_kl_loss(series[u], (
                                prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                    self.win_size)).detach()) * temperature
                        prior_loss += my_kl_loss(
                            (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                    self.win_size)),
                            series[u].detach()) * temperature
                metric = torch.softmax((-series_loss - prior_loss), dim=-1)

                cri = metric * loss
                cri = cri.detach().cpu().numpy()
                attens_energy.append(cri)
                test_labels.append(labels)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        test_labels = np.array(test_labels)

        original_sequences = np.concatenate(original_sequences, axis=0)
        predicted_sequences = np.concatenate(predicted_sequences, axis=0)

        original_sequences2 = np.concatenate(original_sequences2, axis=0)
        predicted_sequences2 = np.concatenate(predicted_sequences2, axis=0)

        # test_energy_series = pd.Series(test_energy)
        # print(test_energy_series.describe())

        thresh_pot, _ = pot(test_energy, risk=0.0195) # control the risk
        print("Threshold_POT :", thresh_pot)

        # pred = (test_energy > thresh).astype(int)
        pred = (test_energy > thresh_pot).astype(int)

        gt = test_labels.astype(int)

        print("pred:   ", pred.shape)
        print("gt:     ", gt.shape)


        pred = self.adjust_detection(pred, gt)



        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, support = precision_recall_fscore_support(gt, pred,
                                                                              average='binary')
        print(
            "Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
                accuracy, precision,
                recall, f_score))
        
        # Original Sequence vs Predicted Sequence Plot
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(y=original_sequences.flatten(), mode='lines', name='Original Sequence', line=dict(color='blue')))
        fig1.add_trace(go.Scatter(y=predicted_sequences.flatten(), mode='lines', name='Predicted Sequence', line=dict(color='orange')))
        fig1.update_layout(title='Original Sequence vs Predicted Sequence (All Batches)',
                        xaxis_title='Time Step',
                        yaxis_title='Value')
        fig1.write_html(f"{self.model_save_path}/{self.dataset}_{self.group}_entire_sequence1_comparison.html")

        # Anomaly Score vs Threshold Plot
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(y=test_energy, mode='markers', name='Anomaly Score', line=dict(color='purple')))
        fig2.add_trace(go.Scatter(y=[thresh]*len(test_energy), mode='lines', name='Threshold', line=dict(color='green', dash='dash')))
        fig2.update_layout(title='Anomaly Score vs Threshold',
                        xaxis_title='Sample Index',
                        yaxis_title='Score')
        fig2.write_html(f"{self.model_save_path}/{self.dataset}_{self.group}_anomaly_score_vs_threshold.html")

        # Ground Truth vs Prediction Plot
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(y=gt, mode='lines', name='Ground Truth', line=dict(color='blue')))
        fig3.add_trace(go.Scatter(y=pred, mode='lines', name='Prediction', line=dict(color='red', dash='dash')))
        fig3.update_layout(title='Ground Truth vs Prediction',
                        xaxis_title='Sample Index',
                        yaxis_title='Value')
        fig3.write_html(f"{self.model_save_path}/{self.dataset}_{self.group}_gt_vs_pred.html")

        # Original Sequence vs Predicted Sequence Plot 2
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(y=original_sequences2.flatten(), mode='lines', name='Original Sequence2', line=dict(color='blue')))
        fig4.add_trace(go.Scatter(y=predicted_sequences2.flatten(), mode='lines', name='Predicted Sequence2', line=dict(color='orange')))
        fig4.update_layout(title='Original Sequence2 vs Predicted Sequence2 (All Batches)',
                        xaxis_title='Time Step',
                        yaxis_title='Value')
        fig4.write_html(f"{self.model_save_path}/{self.dataset}_{self.group}_entire_sequence2_comparison.html")


        # Run POT ETC.

        print("Running POT Anomaly Detection...")
        pot_pred = np.array([])
        pot_gt = np.array([])
        all_test_errors = np.array([])
        all_thresholds = np.array([])  

        for group_name in test_groups:
            temp_pred, temp_gt, temp_test_error, temp_threshold = self.test_pot(group_name,max_energy)

            # Concatenate predictions and ground truth
            pot_pred = np.concatenate((pot_pred, temp_pred))
            pot_gt = np.concatenate((pot_gt, temp_gt))

            # Concatenate test errors
            all_test_errors = np.concatenate((all_test_errors, temp_test_error))

            # Expand threshold to match the length of test_error
            expanded_threshold = np.full(len(temp_test_error), temp_threshold)
            all_thresholds = np.concatenate((all_thresholds, expanded_threshold))

      
        # Metrics for each method
        print("\nComparing Results:\n")

        # POT Metrics
        pot_accuracy = accuracy_score(pot_gt, pot_pred)
        pot_precision, pot_recall, pot_f1, _ = precision_recall_fscore_support(pot_gt, pot_pred, average='binary')
        print(f"POT - Accuracy: {pot_accuracy:.4f}, Precision: {pot_precision:.4f}, Recall: {pot_recall:.4f}, F1-Score: {pot_f1:.4f}")

    
        # # Summary of Results
        results = {
            'DEFAULT': {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1_score': f_score},
            'POT': {'accuracy': pot_accuracy, 'precision': pot_precision, 'recall': pot_recall, 'f1_score': pot_f1},
           }

        results_str = (
            f"DEFAULT - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f_score:.4f}\n"
            f"POT - Accuracy: {pot_accuracy:.4f}, Precision: {pot_precision:.4f}, Recall: {pot_recall:.4f}, F1-Score: {pot_f1:.4f}\n"
          )

        with open(f"{self.model_save_path}/{self.dataset}_{self.group}_output.txt", "w") as f:
            f.write("Baseline threshold : " + str(thresh)+'\n')
            f.write("pred.shape : " + str(pred.shape)+'\n')
            f.write("gt.shape : " + str(gt.shape)+'\n')
            f.write("Results:\n" + results_str)


        fig5 = go.Figure()

        # Scatter plot for anomaly scores
        fig5.add_trace(go.Scatter(y=all_test_errors, mode='markers', name='Anomaly Score', line=dict(color='purple')))

        # Threshold line for all groups
        fig5.add_trace(go.Scatter(y=all_thresholds, mode='lines', name='Threshold',
                                line=dict(color='green', dash='dash')))

        fig5.update_layout(
            title='Anomaly Score vs Threshold (Group-specific Thresholds)',
            xaxis_title='Sample Index',
            yaxis_title='Score',
            plot_bgcolor="#f5f5f5",
            font=dict(color='black')
        )

        # Save the plot
        fig5.write_html(f"{self.model_save_path}/{self.dataset}_anomaly_score_vs_threshold_POT.html")




        return results


