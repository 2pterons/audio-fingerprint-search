import os
import math
import argparse
import pickle
from pydub import AudioSegment
from utils.fingerprint import FingerPrint

parser = argparse.ArgumentParser()
parser.add_argument("mode", choices=["build-db", "cut-query", "cut-multi", "query"])
parser.add_argument("--query", help="Query audio path")
parser.add_argument("--file_name", default="I_fall_in_love_too_easily.mp3", help="오디오 파일 이름")
parser.add_argument("--segment_duration_ms", type=int, default=10000, help="세그먼트 길이(ms)")
parser.add_argument("--segment_duration", type=int, default=10, help="cut-multi 모드에서 자를 segment 길이 (초)")
parser.add_argument("--sr", type=int, default=22050, help="샘플링 레이트")
parser.add_argument("--start_time", type=float, default=120.0, help="쿼리용 잘라낼 시작 시간 (초)")
parser.add_argument("--duration", type=float, default=30.0, help="쿼리용 잘라낼 길이 (초)")
parser.add_argument("--db_path", default="./fingerprint_db.pkl", help="Fingerprint DB 저장경로")

args = parser.parse_args()

fp = FingerPrint()

if args.mode == "build-db":
    print("Fingerprint DB 생성 중...")
    file_paths = [
        os.path.join("../audio_samples", f)
        for f in os.listdir("../audio_samples")
        if f.endswith(".mp3") or f.endswith(".wav")
    ]
    db = fp.build_fingerprint_db(file_paths, segment_duration_ms=args.segment_duration_ms, sr=args.sr)
    
    with open(args.db_path, "wb") as f:
        pickle.dump(db, f)
    print(f"DB 저장 완료: {args.db_path}")

elif args.mode == "cut-query":
    if not args.query:
        # output_path = f"../audio_samples/query_segment.wav"
        output_path = f"../audio_segments/{os.path.splitext(args.file_name)[0]}.wav"
        y, sr = fp.extract_segment(
            os.path.join("../audio_samples", args.file_name),
            start_time=args.start_time,
            duration=args.duration,
            
            sr=args.sr,
            output_path=output_path
        )
        print(f"쿼리 오디오 segment 생성 완료: {output_path}")

elif args.mode == "cut-multi":
    input_path = os.path.join("../audio_samples", args.file_name)
    output_dir = "../audio_segments"
    os.makedirs(output_dir, exist_ok=True)

    audio = AudioSegment.from_file(input_path)
    duration_ms = len(audio)
    seg_len_ms = args.segment_duration * 1000

    num_segments = math.ceil(duration_ms / seg_len_ms)
    print(f"{args.file_name} → 총 {num_segments}개 segment 생성 중...")

    for i in range(num_segments):
        start_ms = i * seg_len_ms
        end_ms = min((i + 1) * seg_len_ms, duration_ms)
        segment = audio[start_ms:end_ms]

        output_path = os.path.join(
            output_dir,
            f"{os.path.splitext(args.file_name)[0]}_{i*args.segment_duration:03.0f}.wav"
        )
        segment.export(output_path, format="wav")
        print(f"저장됨: {output_path}")

elif args.mode == "query":
    if not args.query:
        raise ValueError("쿼리 오디오 파일 경로 (--query)가 필요합니다.")
    if not os.path.exists(args.db_path):
        raise FileNotFoundError(f"DB 파일 없음: {args.db_path}")
    with open(args.db_path, "rb") as f:
        db = pickle.load(f)

    match_file, match_start, match_score = fp.search_query(
        query_audio_path=args.query,
        fp_db=db,
        segment_duration_ms=args.segment_duration_ms,
        sr=args.sr
    )

    print("\n검색 결과:")
    print(f"가장 유사한 곡: {match_file}")
    print(f"세그먼트 시작 시간: {match_start}초")
    print(f"유사도 점수: {match_score}")