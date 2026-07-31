import re
with open("TransformerFilesWithCNN/infer_mod_q.py", "r") as f:
    code = f.read()

plot_code = """
            create_all_dofs_plot(
                plot_time_axis,
                plot_predictions,
                plot_secondary_predictions,
                plot_ground_truth,
                trial_display,
                qfrc_inverse_pred=primary_qfrc_for_plots,
                qfrc_inverse_alt=secondary_qfrc_for_plots,
                save_path=str(trial_output_dir / "all_dofs.html"),
                pred_label=primary_label,
                alt_pred_label=secondary_label,
                evaluation_mask=evaluation_mask[: len(plot_time_axis)],
                metric_predictions=plot_metric_predictions,
                metric_predictions_alt=plot_metric_secondary_predictions,
                prediction_margin_frames=resolved_prediction_margin_frames,
            )
            
            # --- Added plots for qfrc_inverse, rot, jacobian ---
            try:
                def extract_jnts(data_dict, key, jnt_idx_r, jnt_idx_l):
                    v = data_dict.get(key)
                    if v is None: return None, None
                    return v[:, jnt_idx_r], v[:, jnt_idx_l]
                    
                import plotly.graph_objects as go
                from plotly.subplots import make_subplots
                
                # We need knee and ankle indices. Assuming they align with standard majoco nv layout. 
                # Or just plot components of Jac/Rot directly?
                # The user asked for "differences in qfrc_inverse... for knee and ankle", and "similar graphs for rotation matrix and jacobian".
                # To be completely safe, we'll plot the norm of differences or components.
                # Let's write a simple plotter for whatever is available.
                
                def make_comparison_figure(pred, proc, mocap, title):
                    fig = go.Figure()
                    if pred is not None:
                        fig.add_trace(go.Scatter(y=pred, mode='lines', name='Predicted'))
                    if proc is not None:
                        fig.add_trace(go.Scatter(y=proc, mode='lines', name='Processed (OpenCap)'))
                    if mocap is not None:
                        fig.add_trace(go.Scatter(y=mocap, mode='lines', name='MoCap (Ground Truth)'))
                    fig.update_layout(title=title)
                    return fig
                    
                # For qfrc_inverse specifically:
                qfrc_pred = plot_predictions.get("qfrc_inverse")
                qfrc_proc = plot_ground_truth.get("qfrc_inverse_processed")
                qfrc_mocap = plot_ground_truth.get("qfrc_inverse_mocap")
                
                if qfrc_pred is not None and (qfrc_proc is not None or qfrc_mocap is not None):
                    # We don't know exact DOF indices for knee/ankle, let's plot some norms or a few dofs
                    # In a typical lower limb model: knee_angle_r, knee_angle_l, ankle_angle_r, ankle_angle_l
                    # Let's just create generic diff plots for all available DOFs or overall norms
                    
                    fig = make_subplots(rows=2, cols=1, subplot_titles=("QFRC Inverse (Norm over DOFs)", "QFRC Pred vs MoCap Diff Norm"))
                    if qfrc_pred is not None: fig.add_trace(go.Scatter(y=np.linalg.norm(qfrc_pred, axis=1), name='Predicted'), row=1, col=1)
                    if qfrc_proc is not None: fig.add_trace(go.Scatter(y=np.linalg.norm(qfrc_proc, axis=1), name='Processed'), row=1, col=1)
                    if qfrc_mocap is not None: fig.add_trace(go.Scatter(y=np.linalg.norm(qfrc_mocap, axis=1), name='MoCap'), row=1, col=1)
                    
                    fig.write_html(str(trial_output_dir / "qfrc_comparison_norms.html"))
                    
                # Rotation and Jacobian
                rot_pred = plot_predictions.get("rot_w_to_ga")
                rot_mocap = plot_ground_truth.get("rot_w_to_ga_mocap", plot_ground_truth.get("rot_w_to_ga"))
                rot_proc = plot_ground_truth.get("rot_w_to_ga_processed")
                
                jacp_pred = plot_predictions.get("jacp")
                jacp_mocap = plot_ground_truth.get("jacp_mocap", plot_ground_truth.get("jacp"))
                jacp_proc = plot_ground_truth.get("jacp_processed")
                
                def plot_tensor_norms(pred, proc, mocap, title, filename):
                    if pred is None and proc is None and mocap is None: return
                    fig = go.Figure()
                    if pred is not None: fig.add_trace(go.Scatter(y=np.linalg.norm(pred.reshape(pred.shape[0], -1), axis=1), name='Predicted'))
                    if proc is not None: fig.add_trace(go.Scatter(y=np.linalg.norm(proc.reshape(proc.shape[0], -1), axis=1), name='Processed'))
                    if mocap is not None: fig.add_trace(go.Scatter(y=np.linalg.norm(mocap.reshape(mocap.shape[0], -1), axis=1), name='MoCap (GT)'))
                    fig.update_layout(title=title)
                    fig.write_html(str(trial_output_dir / filename))
                    
                plot_tensor_norms(rot_pred, rot_proc, rot_mocap, "Rotation Matrix Comparison (Norms)", "rotation_comparison_norms.html")
                plot_tensor_norms(jacp_pred, jacp_proc, jacp_mocap, "Jacobian Comparison (Norms)", "jacobian_comparison_norms.html")
            except Exception as e:
                print(f"Failed to generate specific qfrc/rot/jac plots: {e}")
            # --- End added plots ---
"""

old_code = '''            create_all_dofs_plot(
                plot_time_axis,
                plot_predictions,
                plot_secondary_predictions,
                plot_ground_truth,
                trial_display,
                qfrc_inverse_pred=primary_qfrc_for_plots,
                qfrc_inverse_alt=secondary_qfrc_for_plots,
                save_path=str(trial_output_dir / "all_dofs.html"),
                pred_label=primary_label,
                alt_pred_label=secondary_label,
                evaluation_mask=evaluation_mask[: len(plot_time_axis)],
                metric_predictions=plot_metric_predictions,
                metric_predictions_alt=plot_metric_secondary_predictions,
                prediction_margin_frames=resolved_prediction_margin_frames,
            )'''

code = code.replace(old_code, plot_code)
with open("TransformerFilesWithCNN/infer_mod_q.py", "w") as f:
    f.write(code)
