import argparse
from models.wavlm_utils import get_wavlm_embedding
from models.clap_utils import get_clap_embedding

def main():
    parser = argparse.ArgumentParser(description="Audio Embedding Search Controller")

    parser.add_argument("--user", required=True)
    parser.add_argument("--model", choices=["clap", "wavlm"], required=True)
    parser.add_argument("--task", choices=["init", "embed", "split", "search"], required=True)
    parser.add_argument("--audio_path", type=str, help="Path to single audio file")
    parser.add_argument("--audio_dir", type=str, help="Directory with audio files")
    parser.add_argument("--query_path", type=str, help="Path to query segment")
    parser.add_argument("--milvus_host", type=str, default="175.125.94.218")
    parser.add_argument("--milvus_port", type=str, default="19530")

    parser.add_argument("--segment_duration", type=int, default=5)

    args = parser.parse_args()

    connections.connect(host=args.milvus_host, port=args.milvus_port)

    if args.task == "init":
        milvus_ops.init_collection(args.model)

    elif args.task == "embed":
        if not args.audio_dir:
            raise ValueError("`--audio_dir` is required for embedding")
        embedding_ops.embed_and_insert(args.user, args.model, args.audio_dir, args.segment_duration)

    elif args.task == "split":
        if not args.audio_path:
            raise ValueError("`--audio_path` is required for splitting")
        split_audio_pydub(args.audio_path)

    elif args.task == "search":
        if not args.query_path:
            raise ValueError("`--query_path` is required for search")
        embedding_ops.search(args.user, args.model, args.query_path)

if __name__ == "__main__":
    main()
