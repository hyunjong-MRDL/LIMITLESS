from TransUNet import TransUNet3D

def build_model(args):
    if args.model == 'TransUNet3D':
        model = TransUNet3D(input_size = args.input_size,
                             filter_num = args.filter_num,
                             out_channel = args.out_channel,
                             patch_size = args.patch_size,
                             patch_stride = args.patch_stride,
                             stack_num_down = args.stack_num_down,
                             stack_num_up = args.stack_num_up,
                             embed_dim = args.embedding_dim,
                             num_mlp = args.num_mlp,
                             num_heads = args.num_heads,
                             num_transformer = args.num_transformer,
                             )
    return model