from util import *
from model import *
from torch.optim import Adam
from sklearn.metrics import precision_recall_fscore_support
import time
import pickle
import gc

assert not os.path.exists(output_root)

all_data, all_label, all_edge_index = load_data(shuffle=shuffle)

print(len(all_data), len(all_label), all_data[0].shape, all_label[0].shape)
in_dim = all_data[0].shape[-1]
out_dim = 5
n_node_in_graph = all_data[0].shape[-2]

training_accs, training_losses = [], []
val_accs, val_losses = [], []
y, y_pred = [], []


def run(train_graph, val_graph, fold):
    fold_training_accs, fold_training_losses = [], []
    fold_val_accs, fold_val_losses = [], []
    fold_ys, fold_y_preds = [], []

    for epoch in range(1, epochs + 1):
        print(f'epoch{epoch}...')
        t_start = time.perf_counter()

        fold_training_acc, fold_training_loss = train(train_graph)
        fold_val_acc, fold_val_loss, fold_y, fold_y_pred = evaluate(val_graph)

        t_end = time.perf_counter()
        print(f'epoch{epoch} time: {t_end - t_start:.2f}s\tacc: {fold_training_acc:.3f}\tloss: {fold_training_loss:.3f}'
              + f'\tval_acc: {fold_val_acc:.3f}\tval_loss: {fold_val_loss:.3f}')

        fold_training_accs.append(fold_training_acc)
        fold_training_losses.append(fold_training_loss)
        fold_val_accs.append(fold_val_acc)
        fold_val_losses.append(fold_val_loss)
        fold_ys.append(fold_y)
        fold_y_preds.append(fold_y_pred)

        print('=' * 20)
        gc.collect()

    training_accs.append(fold_training_accs)
    training_losses.append(fold_training_losses)
    val_accs.append(fold_val_accs)
    val_losses.append(fold_val_losses)
    best_idx = torch.tensor(fold_val_accs).max(0)[1].item()
    global y, y_pred
    y += fold_ys[best_idx]
    y_pred += fold_y_preds[best_idx]

    acc_file = open(os.path.join(criterion_root, f'fold{fold + 1}_acc'), 'wb')
    loss_file = open(os.path.join(criterion_root, f'fold{fold + 1}_loss'), 'wb')
    val_acc_file = open(os.path.join(criterion_root, f'fold{fold + 1}_val_acc'), 'wb')
    val_loss_file = open(os.path.join(criterion_root, f'fold{fold + 1}_val_loss'), 'wb')

    pickle.dump(fold_training_accs, acc_file)
    pickle.dump(fold_training_losses, loss_file)
    pickle.dump(fold_val_accs, val_acc_file)
    pickle.dump(fold_val_losses, val_loss_file)

    acc_file.close()
    loss_file.close()
    val_acc_file.close()
    val_loss_file.close()


def train(train_graph):
    training_acc, training_loss = 0, 0
    n_graph = 0

    n_batch = len(train_graph)
    n_graph += (n_batch - 1) * batch_size + train_graph[-1].num_graphs
    for j in range(n_batch):
        graph = train_graph[j].to(device)

        model.train()
        optimizer.zero_grad()
        out = model(graph).unsqueeze(-1)
        acc = out.max(1)[1].eq(graph.y).sum().item()
        loss = F.nll_loss(out, graph.y, reduction='sum')
        loss_num = loss.detach().item()
        training_acc += acc
        training_loss += loss_num

        loss.backward()
        optimizer.step()

    print('-' * 20)

    training_acc /= n_graph
    training_loss /= n_graph

    return training_acc, training_loss


def evaluate(val_graph):
    model.eval()

    all_acc = 0
    all_loss = 0
    n_graph = 0
    batch_y, batch_y_pred = [], []

    n_batch = len(val_graph)
    n_graph += (n_batch - 1) * batch_size + val_graph[-1].num_graphs
    for j in range(n_batch):
        with torch.no_grad():
            graph = val_graph[j].to(device)
            logits = model(graph).unsqueeze(-1)
            pred = logits.max(1)[1]

        l = graph.y
        loss = F.nll_loss(logits, l, reduction='sum')
        all_acc += pred.eq(l).sum().item()
        all_loss += loss.detach().item()
        batch_y += l.squeeze().tolist()
        batch_y_pred += pred.squeeze().tolist()

    epoch_acc = all_acc / n_graph
    epoch_loss = all_loss / n_graph

    return epoch_acc, epoch_loss, batch_y, batch_y_pred


model = MyModel(in_dim, emb_dim, out_dim, num_type, num_relation, n_node_in_graph=n_node_in_graph)
model.to(device)
optimizer = Adam(model.parameters(), lr=lr, weight_decay=0.001)
model.show_parameter_num()

if not os.path.exists(output_root):
    os.mkdir(output_root)
    os.mkdir(criterion_root)
    os.mkdir(plot_root)

if torch.cuda.is_available():
    torch.cuda.synchronize()

for i in range(k_fold):
    if i not in range(folds_from, folds_to):
        continue
    print(f'=====fold {i + 1}=====')
    model.reset_parameters()

    if i != 0:
        all_data, all_label, all_edge_index = load_data(fold=i, shuffle=shuffle)

    batched_train_graph, batched_val_graph = get_batches(all_data, all_label, all_edge_index, i)

    run(batched_train_graph, batched_val_graph, i)
    del batched_train_graph, batched_val_graph
    torch.cuda.empty_cache()

if torch.cuda.is_available():
    torch.cuda.synchronize()

torch.save(model.state_dict(), os.path.join(output_root, 'model.pt'))

training_accs = np.mean(training_accs, axis=0)
training_losses = np.mean(training_losses, axis=0)
val_accs = np.mean(val_accs, axis=0)
val_losses = np.mean(val_losses, axis=0)

epochs = range(1, len(training_losses) + 1)
plt.plot(epochs, training_accs, 'bo', label='Training acc')
plt.plot(epochs, val_accs, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.savefig(os.path.join(plot_root, f'acc.jpg'))

plt.figure()
plt.plot(epochs, training_losses, 'bo', label='Training loss')
plt.plot(epochs, val_losses, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.savefig(os.path.join(plot_root, f'loss.jpg'))

plt.figure()
plot_Matrix(y, y_pred)

precision_recall_f1 = precision_recall_fscore_support(y, y_pred, average='macro')
precision_recall_f1_file = open(os.path.join(criterion_root, f'precision_recall_f1_{folds_from}_{folds_to}'), 'wb')
pickle.dump(precision_recall_f1, precision_recall_f1_file)
precision_recall_f1_file.close()

y_file = open(os.path.join(criterion_root, f'y_{folds_from + 1}_{folds_to}'), 'wb')
y_pred_file = open(os.path.join(criterion_root, f'y_pred_{folds_from + 1}_{folds_to}'), 'wb')
pickle.dump(y, y_file)
pickle.dump(y_pred, y_pred_file)
y_file.close()
y_pred_file.close()
