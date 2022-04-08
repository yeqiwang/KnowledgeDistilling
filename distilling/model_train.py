import tensorflow as tf

from distilling import train_config as tc
from distilling import data_load, model_sturcture, distilling


def clf_train(model_name: str, model_path, tb_path):
    """
    模型训练
    :return:
    """
    train_ds, test_ds = data_load.get_dataset()

    cosine_lr_schdule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=tc.TRAIN_CONFIG['lr_schedule']['initial_lr'],
        decay_steps=tc.TRAIN_CONFIG['lr_schedule']['decay_step'],
        alpha=tc.TRAIN_CONFIG['lr_schedule']['alpha'],
    )

    if model_name == 'teacher':
        model = model_sturcture.TeacherModel()
    else:
        model = model_sturcture.StudentModel(with_softmax=True)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=cosine_lr_schdule, amsgrad=True),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
        metrics=[
            tf.keras.metrics.CategoricalAccuracy('acc'),
        ]
    )

    my_callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir=tb_path, histogram_freq=1),
        tf.keras.callbacks.EarlyStopping(patience=5, monitor='val_loss', restore_best_weights=True)
    ]

    model.fit(x=train_ds, epochs=tc.TRAIN_CONFIG['epoch'], callbacks=my_callbacks, validation_data=test_ds, verbose=1)

    model.evaluate(train_ds)
    model.evaluate(test_ds)

    model.save_weights(model_path)


def distilling_train(model_name:str, model_path, tb_path):
    train_ds, test_ds = data_load.get_dataset()

    cosine_lr_schdule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=tc.TRAIN_CONFIG['lr_schedule']['initial_lr'],
        decay_steps=tc.TRAIN_CONFIG['lr_schedule']['decay_step'],
        alpha=tc.TRAIN_CONFIG['lr_schedule']['alpha'],
    )

    teacher_model = model_sturcture.TeacherModel()
    teacher_model.load_weights('./cheakpoint/teacher/')
    teacher_model = tf.keras.Model(teacher_model.inputs, teacher_model.get_layer(index=-2).output)

    student_model = model_sturcture.StudentModel(with_softmax=False)

    dist = distilling.Distilling(student_model=student_model, teacher_model=teacher_model)
    dist.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=cosine_lr_schdule, amsgrad=True),
        clf_loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
        kd_loss=tf.keras.losses.KLDivergence(),
        T=2,
        alpha=0.9,
        metrics=[
            tf.keras.metrics.CategoricalAccuracy('acc')
        ]
    )
    my_callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir=tb_path, histogram_freq=1),
        tf.keras.callbacks.EarlyStopping(patience=5, monitor='val_clf_loss', restore_best_weights=True)
    ]
    dist.fit(x=train_ds, epochs=tc.TRAIN_CONFIG['epoch'], callbacks=my_callbacks, validation_data=test_ds, verbose=1)

    dist.evaluate(train_ds)
    dist.evaluate(test_ds)