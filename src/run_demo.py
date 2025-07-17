import os
import argparse
from fingerprint import FingerPrint

parser = argparse.ArgumentParser()
parser.add_argument("mode", choices=["build-db", "query"])
parser.add_argument("--query", help="Query audio path")
parser.add_argument("--file_name", default="I_fall_in_love_too_easily.mp3", help="오디오 파일 이름")
parser.add_argument("--segment_duration_ms", type=int, default=10000, help="세그먼트 길이(ms)")
parser.add_argument("--sr", type=int, default=22050, help="샘플링 레이트")
parser.add_argument("--start_time", type=float, default=80.0, help="쿼리용 잘라낼 시작 시간 (초)")
parser.add_argument("--duration", type=float, default=10.0, help="쿼리용 잘라낼 길이 (초)")
parser.add_argument("--db_path", default="./fingerprint_db.pkl", help="Fingerprint DB 저장경로")

args = parser.parse_args()

fp = FingerPrint()

if args.mode == "build-db":
    print("Fingerprint DB 생성 중...")
    file_paths = [
        os.path.join("./data", f)
        for f in os.listdir("./data")
        if f.endswith(".mp3") or f.endswith(".wav")
    ]
    db = fp.build_fingerprint_db(file_paths, segment_duration_ms=args.segment_duration_ms, sr=args.sr)
    
    with open(args.db_path, "wb") as f:
        pickle.dump(db, f)
    print(f"DB 저장 완료: {args.db_path}")

elif args.mode == "query":
    if not args.query:
        output_path = "./data/query_segment.wav"
        y, sr = extract_segment(
            os.path.join("./data", args.file_name),
            start_time=args.start_time,
            duration=args.duration,
            sr=args.sr,
            output_path=output_path
        )
        print(f"쿼리 오디오 segment 생성 완료: {output_path}")
        query_audio = output_path
    else:
        query_audio = args.query
    
    if not os.path.exists(args.db_path):
        raise FileNotFoundError(f"DB 파일 없음: {args.db_path}")
    with open(args.db_path, "rb") as f:
        db = pickle.load(f)

    match_file, match_start, match_score = fp.search_query(query_audio, db, segment_duration_ms=args.segment_duration_ms, sr=args.sr)

    print("\n검색 결과:")
    print(f"가장 유사한 곡: {match_file}")
    print(f"세그먼트 시작 시간: {match_start}초")
    print(f"유사도 점수: {match_score}")