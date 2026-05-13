from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation, PillowWriter
import io
import base64
from datetime import datetime
import os
import traceback
import json
from dotenv import load_dotenv

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
try:
    import dashscope
    DASHSCOPE_AVAILABLE = True
except ImportError:
    DASHSCOPE_AVAILABLE = False
load_dotenv()
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

app = FastAPI(title="概率学习平台API", version="1.0.0")

# 将第33-39行的 CORS 配置改为:
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# 用户会话数据存储文件
USER_DATA_FILE = "user_sessions.json"

def load_user_data():
    """加载用户数据"""
    if os.path.exists(USER_DATA_FILE):
        try:
            with open(USER_DATA_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_user_data(data):
    """保存用户数据到JSON文件"""
    with open(USER_DATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

class UserLogin(BaseModel):
    name: str
    student_id: str

class ChapterTimeRecord(BaseModel):
    student_id: str
    chapter: str
    start_time: str
    end_time: Optional[str] = None
    duration_seconds: Optional[float] = None

class LoginResponse(BaseModel):
    success: bool
    message: str
    user_info: Optional[Dict[str, str]] = None



class CoinTossRequest(BaseModel):
    n_coin: int = Field(ge=1, le=1000, default=100)


class NeedleRequest(BaseModel):
    n_needle: int = Field(ge=10, le=5000, default=1000)
    L: float = Field(ge=0.1, le=2.0, default=1.0)
    D: float = Field(ge=0.5, le=3.0, default=2.0)


class EventRelationRequest(BaseModel):
    probA: float = Field(ge=0.0, le=1.0, default=0.5)
    probB: float = Field(ge=0.0, le=1.0, default=0.5)
    probAB: float = Field(ge=0.0, le=1.0, default=0.3)


class DiceRollRequest(BaseModel):
    n_dice: int = Field(ge=10, le=1000, default=100)


class GeometricProbRequest(BaseModel):
    a: float = Field(ge=0.1, le=1.0, default=0.5)
    b: float = Field(ge=0.1, le=1.0, default=0.5)


class ConditionalProbRequest(BaseModel):
    pA: float = Field(ge=0.01, le=0.99, default=0.5)
    pB: float = Field(ge=0.01, le=0.99, default=0.5)
    pAB: float = Field(ge=0.01, le=1.0, default=0.25)


class DistributionRequest(BaseModel):
    dist_type: str
    params: Dict[str, float]
    n_samples: Optional[int] = 10000


class CLTRequest(BaseModel):
    dist_type: str = "均匀分布"
    n_samples: int = Field(ge=10, le=1000, default=30)
    n_trials: int = Field(ge=100, le=10000, default=1000)


class LLNRequest(BaseModel):
    dist_type: str = "均匀分布"
    num_trials: int = Field(ge=10, le=1000, default=100)


class SamplingDistRequest(BaseModel):
    mu: float = 0.0
    sigma: float = 1.0
    n: int = Field(ge=5, le=100, default=30)


class OrderStatsRequest(BaseModel):
    dist_type: str = "正态分布"
    sample_size: int = Field(ge=5, le=100, default=20)


class TTestRequest(BaseModel):
    sample_size: int = Field(ge=10, le=1000, default=50)
    hypothesized_mean: float = 0.0
    true_mean: float = 0.0
    alpha: float = Field(ge=0.01, le=0.1, default=0.05)


class TwoSampleTTestRequest(BaseModel):
    sample_size1: int = Field(ge=10, le=1000, default=50)
    sample_size2: int = Field(ge=10, le=1000, default=50)
    mean1: float = 0.0
    mean2: float = 2.0
    std1: float = 1.0
    std2: float = 1.0
    alpha: float = Field(ge=0.01, le=0.1, default=0.05)
    equal_var: bool = True


class ConfidenceIntervalRequest(BaseModel):
    sample_size: int = Field(ge=10, le=1000, default=50)
    confidence_level: float = Field(ge=0.8, le=0.99, default=0.95)


class MomentEstimationRequest(BaseModel):
    n: int = Field(ge=10, le=1000, default=100)


class MLEstimationRequest(BaseModel):
    n: int = Field(ge=10, le=1000, default=100)


class EstimatorEfficiencyRequest(BaseModel):
    dist_type: str = "正态分布"
    sample_size: int = Field(ge=10, le=1000, default=50)


class JointDiscreteRequest(BaseModel):
    prob00: float = 0.25
    prob01: float = 0.25
    prob10: float = 0.25
    prob11: float = 0.25


class JointContinuousRequest(BaseModel):
    dist_type_x: str = "正态分布"
    dist_type_y: str = "正态分布"
    corr_coef: float = 0.0


class AIMessage(BaseModel):
    message: str
    context: Optional[str] = None

class CodeExecutionRequest(BaseModel):
    code: str

def create_gif_animation(frames, duration=100):
    """将多帧图像转换为GIF动图"""
    buf = io.BytesIO()
    frames[0].save(
        buf,
        format='GIF',
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0,
        optimize=True
    )
    buf.seek(0)
    gif_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close('all')
    return f"data:image/gif;base64,{gif_base64}"


def create_plot(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig)
    return f"data:image/png;base64,{img_base64}"


@app.get("/")
async def root():
    return {"message": "概率学习平台API", "version": "1.0.0"}


@app.post("/api/user/login")
async def user_login(request: UserLogin):
    """用户登录"""
    try:
        if not request.name or not request.student_id:
            raise HTTPException(status_code=400, detail="姓名和学号不能为空")

        user_data = load_user_data()

        # 初始化用户数据（如果不存在）
        if request.student_id not in user_data:
            user_data[request.student_id] = {
                "name": request.name,
                "student_id": request.student_id,
                "login_time": datetime.now().isoformat(),
                "chapters": {}
            }
            save_user_data(user_data)

        return {
            "success": True,
            "message": "登录成功",
            "user_info": {
                "name": request.name,
                "student_id": request.student_id
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"登录失败: {str(e)}")


@app.post("/api/user/chapter/start")
async def chapter_start(request: ChapterTimeRecord):
    """记录章节学习开始时间"""
    try:
        user_data = load_user_data()

        if request.student_id not in user_data:
            raise HTTPException(status_code=404, detail="用户未登录")

        # 记录开始时间
        user_data[request.student_id]["chapters"][request.chapter] = {
            "start_time": request.start_time,
            "end_time": None,
            "duration_seconds": None,
            "status": "learning"
        }

        save_user_data(user_data)

        return {
            "success": True,
            "message": "开始时间已记录"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"记录失败: {str(e)}")


@app.post("/api/user/chapter/end")
async def chapter_end(request: ChapterTimeRecord):
    """记录章节学习结束时间"""
    try:
        user_data = load_user_data()

        if request.student_id not in user_data:
            raise HTTPException(status_code=404, detail="用户未登录")

        if request.chapter not in user_data[request.student_id]["chapters"]:
            raise HTTPException(status_code=404, detail="未找到章节开始记录")

        # 计算持续时间
        start_time = datetime.fromisoformat(request.start_time)
        end_time = datetime.fromisoformat(request.end_time)
        duration = (end_time - start_time).total_seconds()

        # 更新结束时间和持续时间
        user_data[request.student_id]["chapters"][request.chapter].update({
            "end_time": request.end_time,
            "duration_seconds": duration,
            "status": "completed"
        })

        save_user_data(user_data)

        return {
            "success": True,
            "message": "结束时间已记录",
            "duration_seconds": duration
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"记录失败: {str(e)}")


@app.get("/api/user/export/{student_id}")
async def export_user_data(student_id: str):
    """导出用户学习数据为Excel"""
    try:
        user_data = load_user_data()

        if student_id not in user_data:
            raise HTTPException(status_code=404, detail="用户不存在")

        user = user_data[student_id]

        # 准备数据
        rows = []
        for chapter, info in user["chapters"].items():
            start_time = info.get("start_time", "")
            end_time = info.get("end_time", "")
            duration = info.get("duration_seconds", 0)

            # 格式化时间
            if start_time:
                start_dt = datetime.fromisoformat(start_time)
                start_str = start_dt.strftime("%Y-%m-%d %H:%M:%S")
            else:
                start_str = ""

            if end_time:
                end_dt = datetime.fromisoformat(end_time)
                end_str = end_dt.strftime("%Y-%m-%d %H:%M:%S")
            else:
                end_str = ""

            # 格式化持续时间
            if duration:
                hours = int(duration // 3600)
                minutes = int((duration % 3600) // 60)
                seconds = int(duration % 60)
                duration_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
            else:
                duration_str = ""

            rows.append({
                "姓名": user["name"],
                "学号": user["student_id"],
                "章节": chapter,
                "开始时间": start_str,
                "结束时间": end_str,
                "学习时长": duration_str,
                "状态": "已完成" if info.get("status") == "completed" else "学习中"
            })

        # 创建DataFrame
        df = pd.DataFrame(rows)

        # 生成Excel文件
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='学习记录')

        output.seek(0)

        # 转换为base64
        excel_base64 = base64.b64encode(output.getvalue()).decode('utf-8')

        return {
            "success": True,
            "filename": f"{user['name']}_{student_id}_学习记录.xlsx",
            "data": excel_base64
        }
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"导出失败: {str(e)}")


@app.get("/api/user/sessions/all")
async def get_all_sessions():
    """获取所有用户的学习数据（用于管理员查看）"""
    try:
        user_data = load_user_data()
        return {
            "success": True,
            "users": user_data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取数据失败: {str(e)}")


@app.post("/api/chapter1/coin-toss")
async def coin_toss(request: CoinTossRequest):
    results = ["正面" if np.random.random() > 0.5 else "反面" for _ in range(request.n_coin)]
    heads = results.count("正面")
    tails = results.count("反面")

    fig, ax = plt.subplots()
    ax.bar(["正面", "反面"], [heads, tails], color=['red', 'blue'])
    ax.set_title("投币结果分布")
    ax.set_ylabel("出现次数")
    img = create_plot(fig)

    return {
        "heads": heads,
        "tails": tails,
        "heads_freq": heads / request.n_coin,
        "tails_freq": tails / request.n_coin,
        "image": img
    }


@app.post("/api/chapter1/coin-toss-animation")
async def coin_toss_animation(request: CoinTossRequest):
    """投币实验动图"""
    try:
        n_frames = min(request.n_coin // 10, 100)
        frame_interval = max(request.n_coin // n_frames, 1)

        fig, ax = plt.subplots(figsize=(10, 6))

        heads_count = 0
        tails_count = 0
        heads_history = []
        tails_history = []

        frames = []

        for i in range(1, request.n_coin + 1):
            if np.random.random() > 0.5:
                heads_count += 1
            else:
                tails_count += 1

            if i % frame_interval == 0 or i == request.n_coin:
                ax.clear()
                categories = ['正面', '反面']
                values = [heads_count, tails_count]
                colors = ['#FF6B6B', '#4ECDC4']

                bars = ax.bar(categories, values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
                ax.set_ylim(0, max(values) * 1.2)
                ax.set_title(f'投币实验 (第 {i} 次)\n正面频率: {heads_count / i:.3f}', fontsize=14, fontweight='bold')
                ax.set_ylabel('出现次数', fontsize=12)
                ax.grid(axis='y', alpha=0.3, linestyle='--')

                for bar, val in zip(bars, values):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width() / 2., height,
                            f'{val}', ha='center', va='bottom', fontsize=11, fontweight='bold')

                buf = io.BytesIO()
                fig.savefig(buf, format='png', bbox_inches='tight', dpi=80)
                buf.seek(0)
                from PIL import Image
                img = Image.open(buf)
                frames.append(img.copy())
                buf.close()

        plt.close(fig)

        if len(frames) > 1:
            gif_image = create_gif_animation(frames[:50], duration=80)
        else:
            gif_image = None

        return {
            "heads": heads_count,
            "tails": tails_count,
            "heads_freq": heads_count / request.n_coin,
            "tails_freq": tails_count / request.n_coin,
            "animation": gif_image,
            "total_frames": len(frames)
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"生成动图失败: {str(e)}")


@app.post("/api/chapter1/needle")
async def buffon_needle(request: NeedleRequest):
    crosses = 0
    for _ in range(request.n_needle):
        y = np.random.uniform(0, request.D / 2)
        theta = np.random.uniform(0, np.pi / 2)
        if y <= (request.L / 2) * np.sin(theta):
            crosses += 1

    if crosses == 0:
        raise HTTPException(status_code=400, detail="没有发生相交情况，请增加投针次数或调整参数")

    pi_estimate = (2 * request.L * request.n_needle) / (request.D * crosses)

    x = np.linspace(0, request.n_needle, request.n_needle)
    y_est = [(2 * request.L * i) / (request.D * max(1, c)) for i, c in enumerate(range(1, request.n_needle + 1), 1)]

    fig, ax = plt.subplots()
    ax.plot(x, y_est, label="估计值")
    ax.axhline(y=np.pi, color='r', linestyle='--', label="真实值")
    ax.set_title("π值估计收敛过程")
    ax.set_xlabel("投针次数")
    ax.set_ylabel("π估计值")
    ax.legend()
    img = create_plot(fig)

    return {
        "crosses": crosses,
        "pi_estimate": pi_estimate,
        "error_rate": abs(pi_estimate - np.pi) / np.pi,
        "image": img
    }


@app.post("/api/chapter1/needle-animation")
async def buffon_needle_animation(request: NeedleRequest):
    """布丰投针动图"""
    try:
        from PIL import Image

        n_display_frames = min(30, request.n_needle // 10)
        frame_interval = max(request.n_needle // n_display_frames, 1)

        crosses = 0
        pi_estimates = []

        frames = []

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        for i in range(1, request.n_needle + 1):
            y = np.random.uniform(0, request.D / 2)
            theta = np.random.uniform(0, np.pi / 2)

            if y <= (request.L / 2) * np.sin(theta):
                crosses += 1

            if i % frame_interval == 0 or i == request.n_needle:
                pi_est = (2 * request.L * i) / (request.D * max(1, crosses))
                pi_estimates.append(pi_est)

                ax1.clear()
                ax1.set_xlim(0, request.D)
                ax1.set_ylim(0, request.D / 2)
                for _ in range(min(i, 100)):
                    y_rand = np.random.uniform(0, request.D / 2)
                    theta_rand = np.random.uniform(0, np.pi / 2)
                    x_start = np.random.uniform(0, request.D)
                    x_end = x_start + (request.L / 2) * np.cos(theta_rand)
                    y_end = y_rand - (request.L / 2) * np.sin(theta_rand)
                    crosses_line = y_end < 0
                    ax1.plot([x_start, x_end], [y_rand, max(0, y_end)],
                             'r-' if crosses_line else 'b-', alpha=0.3, linewidth=0.5)

                for j in range(int(request.D / request.D)):
                    ax1.axhline(y=j * request.D, color='gray', linestyle='--', alpha=0.5)

                ax1.set_title(f'布丰投针模拟 (第 {i} 次)', fontsize=12, fontweight='bold')
                ax1.set_xlabel('X坐标')
                ax1.set_ylabel('Y坐标')
                ax1.grid(True, alpha=0.3)

                ax2.clear()
                x_vals = list(range(1, len(pi_estimates) + 1))
                ax2.plot(x_vals, pi_estimates, 'b-', linewidth=2, label='π估计值')
                ax2.axhline(y=np.pi, color='r', linestyle='--', linewidth=2, label=f'真实值 π={np.pi:.4f}')
                ax2.set_title(f'π值收敛过程\n当前估计: {pi_est:.4f}', fontsize=12, fontweight='bold')
                ax2.set_xlabel('投针批次')
                ax2.set_ylabel('π估计值')
                ax2.legend(loc='best')
                ax2.grid(True, alpha=0.3)

                fig.tight_layout()

                buf = io.BytesIO()
                fig.savefig(buf, format='png', bbox_inches='tight', dpi=80)
                buf.seek(0)
                img = Image.open(buf)
                frames.append(img.copy())
                buf.close()

        plt.close(fig)

        if len(frames) > 1:
            gif_image = create_gif_animation(frames[:30], duration=100)
        else:
            gif_image = None

        pi_estimate = (2 * request.L * request.n_needle) / (request.D * max(1, crosses))

        return {
            "crosses": crosses,
            "pi_estimate": pi_estimate,
            "error_rate": abs(pi_estimate - np.pi) / np.pi,
            "animation": gif_image,
            "total_frames": len(frames)
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"生成动图失败: {str(e)}")

@app.post("/api/chapter1/event-relation")
async def event_relation(request: EventRelationRequest):
    union_prob = request.probA + request.probB - request.probAB
    diff_prob = max(0, request.probA - request.probB)
    cond_prob = request.probAB / request.probA if request.probA > 0 else 0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.set_title(f"事件A∪B的概率: {union_prob:.2f}")
    circle1 = plt.Circle((0.3, 0.5), 0.3, alpha=0.5, color='blue')
    circle2 = plt.Circle((0.7, 0.5), 0.3, alpha=0.5, color='red')
    ax1.add_patch(circle1)
    ax1.add_patch(circle2)
    ax1.text(0.1, 0.5, "A", fontsize=12)
    ax1.text(0.9, 0.5, "B", fontsize=12)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis('off')

    ax2.set_title(f"事件A-B的概率: {diff_prob:.2f}")
    circle1 = plt.Circle((0.3, 0.5), 0.3, alpha=0.5, color='blue')
    circle2 = plt.Circle((0.7, 0.5), 0.3, alpha=0.5, color='white', edgecolor='red')
    ax2.add_patch(circle1)
    ax2.add_patch(circle2)
    ax2.text(0.1, 0.5, "A", fontsize=12)
    ax2.text(0.9, 0.5, "B", fontsize=12)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')

    img = create_plot(fig)

    return {
        "probA": request.probA,
        "probB": request.probB,
        "probAB": request.probAB,
        "union_prob": union_prob,
        "diff_prob": diff_prob,
        "cond_prob": cond_prob,
        "image": img
    }


@app.post("/api/chapter1/dice-roll")
async def dice_roll(request: DiceRollRequest):
    results = [np.random.randint(1, 7) for _ in range(request.n_dice)]
    counts = [results.count(i) for i in range(1, 7)]

    fig, ax = plt.subplots()
    ax.bar(range(1, 7), counts, color='skyblue')
    ax.set_title("骰子投掷结果分布")
    ax.set_xlabel("骰子点数")
    ax.set_ylabel("出现次数")
    img = create_plot(fig)

    return {
        "counts": counts,
        "frequencies": [c / request.n_dice for c in counts],
        "image": img
    }


@app.post("/api/chapter1/geometric")
async def geometric_prob(request: GeometricProbRequest):
    prob = request.a * request.b

    fig, ax = plt.subplots()
    from matplotlib.patches import Rectangle
    ax.add_patch(Rectangle((0, 0), 1, 1, fill=False, edgecolor='black'))
    ax.add_patch(Rectangle((0, 0), request.a, request.b, fill=True, color='blue', alpha=0.5))
    ax.text(request.a / 2, request.b / 2, f"P = {prob:.2f}", ha='center', va='center')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.set_title("几何概型面积演示")
    img = create_plot(fig)

    return {
        "probability": prob,
        "image": img
    }


@app.post("/api/chapter1/conditional-prob")
async def conditional_prob(request: ConditionalProbRequest):
    pBA = request.pAB / request.pA
    pAB_indep = request.pA * request.pB
    is_independent = abs(request.pAB - pAB_indep) < 0.01
    union_prob = request.pA + request.pB - request.pAB

    fig, ax = plt.subplots()
    labels = ["P(A)", "P(B)", "P(A∩B)"]
    values = [request.pA, request.pB, request.pAB]
    ax.bar(labels, values, color=['blue', 'green', 'red'])
    ax.set_title("概率分布")
    ax.set_ylim(0, 1)
    img = create_plot(fig)

    return {
        "pBA": pBA,
        "pAB_indep": pAB_indep,
        "is_independent": is_independent,
        "union_prob": union_prob,
        "image": img
    }


@app.post("/api/chapter2/distribution")
async def distribution_plot(request: DistributionRequest):
    dist_type = request.dist_type
    params = request.params

    if dist_type == "两点分布":
        p = params.get('p', 0.5)
        x = [0, 1]
        probs = [1 - p, p]
        fig, ax = plt.subplots()
        ax.bar(x, probs, color='skyblue')
        ax.set_title(f"两点分布 PMF (p = {p:.2f})")
        ax.set_xlabel("X")
        ax.set_ylabel("概率")
        ax.set_xticks(x)
        ax.set_ylim(0, 1)

    elif dist_type == "二项分布":
        n = int(params.get('n', 10))
        p = params.get('p', 0.5)
        k = np.arange(0, n + 1)
        cdf = np.cumsum(stats.binom.pmf(k, n, p))
        fig, ax = plt.subplots()
        ax.step(k, cdf, where='post')
        ax.set_title(f"二项分布CDF (n = {n}, p = {p:.2f})")
        ax.set_xlabel("成功次数k")
        ax.set_ylabel("累积概率")
        ax.set_xticks(k)
        ax.set_ylim(0, 1.1)

    elif dist_type == "泊松分布":
        lam = params.get('lam', 5.0)
        x = np.arange(0, 31)
        probs = stats.poisson.pmf(x, lam)
        fig, ax = plt.subplots()
        ax.bar(x, probs, color='skyblue')
        ax.set_title(f"泊松分布 PMF (λ = {lam:.2f})")
        ax.set_xlabel("事件发生次数")
        ax.set_ylabel("概率")

    elif dist_type == "正态分布":
        mean = params.get('mean', 0.0)
        std = params.get('std', 1.0)
        x = np.linspace(mean - 3 * std, mean + 3 * std, 100)
        y = stats.norm.pdf(x, mean, std)
        fig, ax = plt.subplots()
        ax.plot(x, y)
        ax.fill_between(x, y, alpha=0.3)
        ax.set_title(f"正态分布 PDF (μ = {mean}, σ = {std})")
        ax.set_xlabel("X")
        ax.set_ylabel("概率密度")

    elif dist_type == "均匀分布":
        min_val = params.get('min_val', 0.0)
        max_val = params.get('max_val', 1.0)
        x = np.linspace(min_val - 1, max_val + 1, 100)
        y = stats.uniform.pdf(x, loc=min_val, scale=max_val - min_val)
        fig, ax = plt.subplots()
        ax.plot(x, y)
        ax.fill_between(x, y, alpha=0.3)
        ax.set_title(f"均匀分布 PDF (min = {min_val}, max = {max_val})")
        ax.set_xlabel("X")
        ax.set_ylabel("概率密度")

    elif dist_type == "指数分布":
        rate = params.get('rate', 1.0)
        x = np.linspace(0, 5, 100)
        y = stats.expon.pdf(x, scale=1 / rate)
        fig, ax = plt.subplots()
        ax.plot(x, y)
        ax.fill_between(x, y, alpha=0.3)
        ax.set_title(f"指数分布 PDF (λ = {rate:.2f})")
        ax.set_xlabel("X")
        ax.set_ylabel("概率密度")

    else:
        raise HTTPException(status_code=400, detail="不支持的分布类型")

    img = create_plot(fig)
    return {"image": img}


@app.post("/api/chapter2/approximation")
async def distribution_approximation(request: Dict[str, Any]):
    approx_type = request.get('type', 'poisson')

    if approx_type == 'poisson':
        n = int(request.get('n', 100))
        p = request.get('p', 0.05)
        lam = n * p
        x = np.arange(0, min(n, int(lam * 3) + 1))
        binom_probs = stats.binom.pmf(x, n, p)
        poisson_probs = stats.poisson.pmf(x, lam)

        fig, ax = plt.subplots()
        ax.plot(x, binom_probs, 'b-', label="二项分布")
        ax.plot(x, poisson_probs, 'r--', label="泊松分布")
        ax.set_title(f"二项分布与泊松分布比较 (λ = {lam:.2f})")
        ax.set_xlabel("成功次数k")
        ax.set_ylabel("概率")
        ax.legend()

    else:
        n = int(request.get('n', 50))
        p = request.get('p', 0.5)
        mean = n * p
        std = np.sqrt(n * p * (1 - p))
        x = np.arange(0, n + 1)
        binom_probs = stats.binom.pmf(x, n, p)
        normal_probs = stats.norm.pdf(x, mean, std)

        fig, ax = plt.subplots()
        ax.bar(x, binom_probs, alpha=0.6, label="二项分布")
        ax.plot(x, normal_probs, 'r-', label="正态分布")
        ax.set_title(f"二项分布与正态分布比较 (n = {n}, p = {p:.2f})")
        ax.set_xlabel("成功次数k")
        ax.set_ylabel("概率")
        ax.legend()

    img = create_plot(fig)
    return {"image": img}


@app.post("/api/chapter3/joint-discrete")
async def joint_discrete(request: JointDiscreteRequest):
    total = request.prob00 + request.prob01 + request.prob10 + request.prob11
    p00 = request.prob00 / total
    p01 = request.prob01 / total
    p10 = request.prob10 / total
    p11 = request.prob11 / total

    marginal_x_0 = p00 + p01
    marginal_x_1 = p10 + p11
    marginal_y_0 = p00 + p10
    marginal_y_1 = p01 + p11

    cond_prob_y1_x0 = p01 / (p00 + p01) if (p00 + p01) > 0 else 0
    cond_prob_y0_x1 = p10 / (p10 + p11) if (p10 + p11) > 0 else 0

    fig1, ax1 = plt.subplots(figsize=(6, 5))
    heat_data = np.array([[p00, p01], [p10, p11]])
    im = ax1.imshow(heat_data, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax1.text(j, i, f"{heat_data[i, j]:.2f}", ha="center", va="center", color="black")
    ax1.set_xticks([0, 1])
    ax1.set_yticks([0, 1])
    ax1.set_xticklabels(["Y=0", "Y=1"])
    ax1.set_yticklabels(["X=0", "X=1"])
    ax1.set_title("联合分布 P(X,Y)")
    fig1.colorbar(im)
    img1 = create_plot(fig1)

    fig2, ax2 = plt.subplots()
    ax2.bar([0, 1], [marginal_x_0, marginal_x_1], color="skyblue")
    ax2.set_title("X的边缘分布")
    ax2.set_xticks([0, 1])
    ax2.set_ylim(0, 1)
    img2 = create_plot(fig2)

    fig3, ax3 = plt.subplots()
    ax3.bar([0, 1], [marginal_y_0, marginal_y_1], color="lightgreen")
    ax3.set_title("Y的边缘分布")
    ax3.set_xticks([0, 1])
    ax3.set_ylim(0, 1)
    img3 = create_plot(fig3)

    return {
        "joint_probs": [[p00, p01], [p10, p11]],
        "marginal_x": [marginal_x_0, marginal_x_1],
        "marginal_y": [marginal_y_0, marginal_y_1],
        "cond_prob_y1_x0": cond_prob_y1_x0,
        "cond_prob_y0_x1": cond_prob_y0_x1,
        "joint_image": img1,
        "marginal_x_image": img2,
        "marginal_y_image": img3
    }


@app.post("/api/chapter3/joint-continuous")
async def joint_continuous(request: JointContinuousRequest):
    n = 1000

    if request.dist_type_x == "正态分布":
        if request.dist_type_y == "正态分布" and request.corr_coef != 0:
            mean = [0, 0]
            cov = [[1, request.corr_coef], [request.corr_coef, 1]]
            x, y = np.random.multivariate_normal(mean, cov, n).T
        else:
            x = np.random.normal(0, 1, n)
    elif request.dist_type_x == "均匀分布":
        x = np.random.uniform(-2, 2, n)
    else:
        x = np.random.exponential(1, n)

    if request.dist_type_y == "正态分布" and not (request.dist_type_x == "正态分布" and request.corr_coef != 0):
        y = np.random.normal(0, 1, n)
    elif request.dist_type_y == "均匀分布":
        y = np.random.uniform(-2, 2, n)
    else:
        y = np.random.exponential(1, n)

    fig1, ax1 = plt.subplots(figsize=(8, 6))
    ax1.scatter(x, y, alpha=0.5)
    ax1.set_title(f"联合分布散点图 (相关系数: {request.corr_coef:.2f})")
    ax1.set_xlabel("X值")
    ax1.set_ylabel("Y值")
    img1 = create_plot(fig1)

    fig2, ax2 = plt.subplots()
    ax2.hist(x, bins=30, color="skyblue", density=True)
    ax2.set_title(f"X的边缘分布 ({request.dist_type_x})")
    ax2.set_xlabel("X值")
    ax2.set_ylabel("密度")
    img2 = create_plot(fig2)

    fig3, ax3 = plt.subplots()
    ax3.hist(y, bins=30, color="lightgreen", density=True)
    ax3.set_title(f"Y的边缘分布 ({request.dist_type_y})")
    ax3.set_xlabel("Y值")
    ax3.set_ylabel("密度")
    img3 = create_plot(fig3)

    sample_corr = np.corrcoef(x, y)[0, 1] if n > 1 else 0

    return {
        "sample_corr": sample_corr,
        "scatter_image": img1,
        "marginal_x_image": img2,
        "marginal_y_image": img3
    }


@app.post("/api/chapter4/digital-features")
async def digital_features(request: DistributionRequest):
    dist_type = request.dist_type
    params = request.params
    n_samples = request.n_samples

    if dist_type == "正态分布":
        mean = params.get('mean', 0.0)
        std = params.get('std', 1.0)
        data = np.random.normal(mean, std, n_samples)
        theoretical_mean = mean
        theoretical_var = std ** 2
        theoretical_std = std
        theoretical_skew = 0
        theoretical_kurt = 0

    elif dist_type == "均匀分布":
        min_val = params.get('min_val', 0.0)
        max_val = params.get('max_val', 1.0)
        data = np.random.uniform(min_val, max_val, n_samples)
        theoretical_mean = (min_val + max_val) / 2
        theoretical_var = (max_val - min_val) ** 2 / 12
        theoretical_std = np.sqrt(theoretical_var)
        theoretical_skew = 0
        theoretical_kurt = -1.2

    elif dist_type == "泊松分布":
        lam = params.get('lam', 5.0)
        data = np.random.poisson(lam, n_samples)
        theoretical_mean = lam
        theoretical_var = lam
        theoretical_std = np.sqrt(lam)
        theoretical_skew = 1 / np.sqrt(lam)
        theoretical_kurt = 1 / lam

    elif dist_type == "指数分布":
        rate = params.get('rate', 1.0)
        data = np.random.exponential(1 / rate, n_samples)
        theoretical_mean = 1 / rate
        theoretical_var = 1 / (rate ** 2)
        theoretical_std = 1 / rate
        theoretical_skew = 2
        theoretical_kurt = 6

    elif dist_type == "二项分布":
        n = int(params.get('n', 10))
        p = params.get('p', 0.5)
        data = np.random.binomial(n, p, n_samples)
        theoretical_mean = n * p
        theoretical_var = n * p * (1 - p)
        theoretical_std = np.sqrt(theoretical_var)
        theoretical_skew = (1 - 2 * p) / np.sqrt(n * p * (1 - p)) if (n * p * (1 - p)) > 0 else 0
        theoretical_kurt = (1 - 6 * p * (1 - p)) / (n * p * (1 - p)) if (n * p * (1 - p)) > 0 else 0
    else:
        raise HTTPException(status_code=400, detail="不支持的分布类型")

    sample_mean = np.mean(data)
    sample_var = np.var(data, ddof=1)
    sample_std = np.std(data, ddof=1)
    sample_skew = stats.skew(data)
    sample_kurt = stats.kurtosis(data)

    fig, ax = plt.subplots()
    ax.hist(data, bins=30, density=True, alpha=0.6, color='blue')
    ax.set_title(f"{dist_type} 分布")
    img = create_plot(fig)

    return {
        "theoretical": {
            "mean": theoretical_mean,
            "var": theoretical_var,
            "std": theoretical_std,
            "skew": theoretical_skew,
            "kurt": theoretical_kurt
        },
        "sample": {
            "mean": sample_mean,
            "var": sample_var,
            "std": sample_std,
            "skew": sample_skew,
            "kurt": sample_kurt
        },
        "image": img
    }


@app.post("/api/chapter5/clt")
async def central_limit_theorem(request: CLTRequest):
    sample_means = []

    for _ in range(request.n_trials):
        if request.dist_type == "均匀分布":
            sample = np.random.uniform(0, 1, request.n_samples)
        elif request.dist_type == "二项分布":
            sample = np.random.binomial(10, 0.2, request.n_samples)
        elif request.dist_type == "泊松分布":
            sample = np.random.poisson(2, request.n_samples)
        else:
            sample = np.random.exponential(1, request.n_samples)
        sample_means.append(np.mean(sample))

    mean = np.mean(sample_means)
    std = np.std(sample_means)

    fig, ax = plt.subplots()
    ax.hist(sample_means, bins=30, density=True, alpha=0.6, color='blue')
    x = np.linspace(mean - 3 * std, mean + 3 * std, 100)
    ax.plot(x, stats.norm.pdf(x, mean, std), 'r-', linewidth=2)
    ax.set_title(f"样本均值分布 (总体: {request.dist_type}, 样本量: {request.n_samples})")
    ax.set_xlabel("样本均值")
    ax.set_ylabel("密度")
    img = create_plot(fig)

    return {
        "mean": mean,
        "std": std,
        "image": img
    }


@app.post("/api/chapter5/clt-animation")
async def clt_animation(request: CLTRequest):
    """中心极限定理动图"""
    try:
        from PIL import Image

        n_frames = 40
        sample_means_all = []

        fig, ax = plt.subplots(figsize=(12, 7))
        frames = []

        for frame_idx in range(1, n_frames + 1):
            current_samples = int(frame_idx * request.n_trials / n_frames)
            sample_means = []

            for _ in range(current_samples):
                if request.dist_type == "均匀分布":
                    sample = np.random.uniform(0, 1, request.n_samples)
                elif request.dist_type == "二项分布":
                    sample = np.random.binomial(10, 0.2, request.n_samples)
                elif request.dist_type == "泊松分布":
                    sample = np.random.poisson(2, request.n_samples)
                else:
                    sample = np.random.exponential(1, request.n_samples)
                sample_means.append(np.mean(sample))

            sample_means_all.extend(sample_means)

            ax.clear()
            ax.hist(sample_means_all, bins=40, density=True, alpha=0.6,
                    color='#667eea', edgecolor='black', linewidth=0.5, label='样本均值分布')

            mean_val = np.mean(sample_means_all)
            std_val = np.std(sample_means_all)

            x = np.linspace(mean_val - 4 * std_val, mean_val + 4 * std_val, 200)
            ax.plot(x, stats.norm.pdf(x, mean_val, std_val),
                    'r-', linewidth=2.5, label='正态拟合')

            ax.set_title(f'中心极限定理演示\n总体: {request.dist_type}, 样本量: {request.n_samples}\n'
                         f'抽样次数: {current_samples}, 均值: {mean_val:.3f}, 标准差: {std_val:.3f}',
                         fontsize=13, fontweight='bold')
            ax.set_xlabel('样本均值', fontsize=12)
            ax.set_ylabel('密度', fontsize=12)
            ax.legend(loc='best', fontsize=10)
            ax.grid(True, alpha=0.3, linestyle='--')

            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight', dpi=80)
            buf.seek(0)
            img = Image.open(buf)
            frames.append(img.copy())
            buf.close()

        plt.close(fig)

        if len(frames) > 1:
            gif_image = create_gif_animation(frames, duration=120)
        else:
            gif_image = None

        return {
            "mean": np.mean(sample_means_all),
            "std": np.std(sample_means_all),
            "animation": gif_image,
            "total_frames": len(frames)
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"生成动图失败: {str(e)}")

@app.post("/api/chapter5/lln")
async def law_large_numbers(request: LLNRequest):
    means = []
    errors = []

    for i in range(1, request.num_trials + 1):
        if request.dist_type == "均匀分布":
            data = np.random.uniform(0, 1, i)
            expected_mean = 0.5
        elif request.dist_type == "二项分布":
            data = np.random.binomial(10, 0.5, i)
            expected_mean = 5
        else:
            data = np.random.poisson(5, i)
            expected_mean = 5

        current_mean = np.mean(data)
        means.append(current_mean)
        errors.append(abs(current_mean - expected_mean))

    fig, ax1 = plt.subplots()
    ax1.set_xlabel('试验次数')
    ax1.set_ylabel('样本均值', color='tab:blue')
    ax1.plot(range(1, request.num_trials + 1), means, color='tab:blue')
    ax1.axhline(y=expected_mean, color='r', linestyle='--', label="期望值")
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()
    ax2.set_ylabel('估计误差', color='tab:red')
    ax2.plot(range(1, request.num_trials + 1), errors, color='tab:red', alpha=0.5)
    ax2.tick_params(axis='y', labelcolor='tab:red')

    fig.tight_layout()
    plt.title(f"大数定律演示 (分布: {request.dist_type})")
    img = create_plot(fig)

    return {
        "expected_mean": expected_mean,
        "final_mean": means[-1],
        "image": img
    }


@app.post("/api/chapter5/lln-animation")
async def law_large_numbers_animation(request: LLNRequest):
    """大数定律动图"""
    try:
        from PIL import Image

        n_frames = 40
        frame_interval = max(request.num_trials // n_frames, 1)

        means = []
        errors = []
        frames = []

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        for i in range(1, request.num_trials + 1):
            if request.dist_type == "均匀分布":
                data_point = np.random.uniform(0, 1)
                expected_mean = 0.5
            elif request.dist_type == "二项分布":
                data_point = np.random.binomial(10, 0.5)
                expected_mean = 5
            else:
                data_point = np.random.poisson(5)
                expected_mean = 5

            means.append((means[-1] * (i - 1) + data_point) / i if i > 1 else data_point)
            errors.append(abs(means[-1] - expected_mean))

            if i % frame_interval == 0 or i == request.num_trials:
                ax1.clear()
                x_vals = list(range(1, len(means) + 1))
                ax1.plot(x_vals, means, 'b-', linewidth=2, label='样本均值')
                ax1.axhline(y=expected_mean, color='r', linestyle='--',
                            linewidth=2.5, label=f'期望值 {expected_mean}')
                ax1.set_title(f'大数定律演示 - 样本均值收敛\n分布: {request.dist_type}',
                              fontsize=13, fontweight='bold')
                ax1.set_xlabel('试验次数', fontsize=11)
                ax1.set_ylabel('样本均值', fontsize=11)
                ax1.legend(loc='best')
                ax1.grid(True, alpha=0.3)

                ax2.clear()
                ax2.plot(x_vals, errors, 'orange', linewidth=2, label='估计误差')
                ax2.fill_between(x_vals, errors, alpha=0.3, color='orange')
                ax2.set_title('误差变化', fontsize=12, fontweight='bold')
                ax2.set_xlabel('试验次数', fontsize=11)
                ax2.set_ylabel('绝对误差', fontsize=11)
                ax2.legend(loc='best')
                ax2.grid(True, alpha=0.3)

                fig.tight_layout()

                buf = io.BytesIO()
                fig.savefig(buf, format='png', bbox_inches='tight', dpi=80)
                buf.seek(0)
                img = Image.open(buf)
                frames.append(img.copy())
                buf.close()

        plt.close(fig)

        if len(frames) > 1:
            gif_image = create_gif_animation(frames, duration=100)
        else:
            gif_image = None

        return {
            "expected_mean": expected_mean,
            "final_mean": means[-1],
            "animation": gif_image,
            "total_frames": len(frames)
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500,detail=f"生成动图失败: {str(e)}")

@app.post("/api/chapter6/sampling-dist")
async def sampling_distribution(request: SamplingDistRequest):
    num_simulations = 1000
    sample_means = []
    sample_vars = []

    for _ in range(num_simulations):
        sample = np.random.normal(request.mu, request.sigma, request.n)
        sample_means.append(np.mean(sample))
        sample_vars.append(np.var(sample, ddof=1))

    fig1, ax1 = plt.subplots()
    ax1.hist(sample_means, bins=30, alpha=0.7, color='blue')
    ax1.axvline(x=np.mean(sample_means), color='red', linestyle='dashed',
                label=f"均值: {np.mean(sample_means):.4f}")
    ax1.set_title("样本均值的分布")
    ax1.set_xlabel("样本均值")
    ax1.set_ylabel("频率")
    ax1.legend()
    img1 = create_plot(fig1)

    fig2, ax2 = plt.subplots()
    ax2.hist(sample_vars, bins=30, alpha=0.7, color='green')
    ax2.axvline(x=np.mean(sample_vars), color='red', linestyle='dashed',
                label=f"均值: {np.mean(sample_vars):.4f}")
    ax2.set_title("样本方差的分布")
    ax2.set_xlabel("样本方差")
    ax2.set_ylabel("频率")
    ax2.legend()
    img2 = create_plot(fig2)

    return {
        "mean_expectation": {
            "theoretical": request.mu,
            "actual": np.mean(sample_means)
        },
        "mean_variance": {
            "theoretical": request.sigma ** 2 / request.n,
            "actual": np.var(sample_means)
        },
        "var_expectation": {
            "theoretical": request.sigma ** 2,
            "actual": np.mean(sample_vars)
        },
        "mean_image": img1,
        "var_image": img2
    }


@app.post("/api/chapter6/order-stats")
async def order_statistics(request: OrderStatsRequest):
    num_samples = 1000
    mins = []
    maxs = []
    medians = []

    for _ in range(num_samples):
        if request.dist_type == "正态分布":
            sample = np.random.normal(0, 1, request.sample_size)
        elif request.dist_type == "均匀分布":
            sample = np.random.uniform(0, 1, request.sample_size)
        else:
            sample = np.random.exponential(1, request.sample_size)

        ordered = np.sort(sample)
        mins.append(ordered[0])
        maxs.append(ordered[-1])
        medians.append(np.median(ordered))

    fig1, ax1 = plt.subplots()
    ax1.hist(mins, bins=30, alpha=0.7, color='blue')
    ax1.set_title("最小值的分布")
    ax1.set_xlabel("最小值")
    ax1.set_ylabel("频率")
    img1 = create_plot(fig1)

    fig2, ax2 = plt.subplots()
    ax2.hist(medians, bins=30, alpha=0.7, color='green')
    ax2.set_title("中位数的分布")
    ax2.set_xlabel("中位数")
    ax2.set_ylabel("频率")
    img2 = create_plot(fig2)

    fig3, ax3 = plt.subplots()
    ax3.hist(maxs, bins=30, alpha=0.7, color='red')
    ax3.set_title("最大值的分布")
    ax3.set_xlabel("最大值")
    ax3.set_ylabel("频率")
    img3 = create_plot(fig3)

    return {
        "stats": {
            "min": {"mean": np.mean(mins), "std": np.std(mins)},
            "median": {"mean": np.mean(medians), "std": np.std(medians)},
            "max": {"mean": np.mean(maxs), "std": np.std(maxs)}
        },
        "min_image": img1,
        "median_image": img2,
        "max_image": img3
    }


@app.post("/api/chapter6/common-distributions")
async def common_distributions(request: Dict[str, Any]):
    dist_name = request.get('dist', 't')

    if dist_name == 't':
        df = request.get('df', 5)
        x = np.linspace(-4, 4, 100)
        y = stats.t.pdf(x, df=df)
        y_norm = stats.norm.pdf(x, 0, 1)
        fig, ax = plt.subplots()
        ax.plot(x, y, label=f"t分布 (df={df})")
        ax.plot(x, y_norm, 'r--', label="标准正态分布")
        ax.set_title(f"t分布与正态分布比较")
        ax.set_xlabel("x")
        ax.set_ylabel("概率密度")
        ax.legend()

    elif dist_name == 'f':
        df1 = request.get('df1', 5)
        df2 = request.get('df2', 5)
        x = np.linspace(0, 5, 100)
        y = stats.f.pdf(x, df1, df2)
        fig, ax = plt.subplots()
        ax.plot(x, y)
        ax.set_title(f"F分布 (df1={df1}, df2={df2})")
        ax.set_xlabel("x")
        ax.set_ylabel("概率密度")

    else:
        df = request.get('df', 5)
        x = np.linspace(0, 20, 100)
        y = stats.chi2.pdf(x, df=df)
        fig, ax = plt.subplots()
        ax.plot(x, y)
        ax.set_title(f"卡方分布 (自由度 = {df})")
        ax.set_xlabel("x")
        ax.set_ylabel("概率密度")

    img = create_plot(fig)
    return {"image": img}


@app.post("/api/chapter7/moment-estimation")
async def moment_estimation(request: MomentEstimationRequest):
    a_true = 0
    b_true = 1
    data = np.random.uniform(a_true, b_true, request.n)

    mu1 = np.mean(data)
    mu2 = np.mean(data ** 2)
    a_hat = mu1 - np.sqrt(3 * (mu2 - mu1 ** 2))
    b_hat = mu1 + np.sqrt(3 * (mu2 - mu1 ** 2))

    fig, ax = plt.subplots()
    ax.hist(data, bins=30, alpha=0.6, color='blue')
    ax.axvline(x=a_hat, color='red', linestyle='dashed', label=f"a估计值: {a_hat:.4f}")
    ax.axvline(x=b_hat, color='green', linestyle='dashed', label=f"b估计值: {b_hat:.4f}")
    ax.axvline(x=a_true, color='black', linestyle='-', alpha=0.3, label=f"真实a: {a_true}")
    ax.axvline(x=b_true, color='black', linestyle='-', alpha=0.3, label=f"真实b: {b_true}")
    ax.set_title(f"均匀分布矩估计 (样本量: {request.n})")
    ax.set_xlabel("样本值")
    ax.set_ylabel("频率")
    ax.legend()
    img = create_plot(fig)

    return {
        "true_values": {"a": a_true, "b": b_true},
        "estimates": {"a": a_hat, "b": b_hat},
        "errors": {"a": abs(a_hat - a_true), "b": abs(b_hat - b_true)},
        "image": img
    }


@app.post("/api/chapter7/mle")
async def maximum_likelihood(request: MLEstimationRequest):
    mu_true = 5
    sigma_true = 2
    data = np.random.normal(mu_true, sigma_true, request.n)

    mu_hat = np.mean(data)
    sigma_hat_mle = np.sqrt(np.mean((data - mu_hat) ** 2))
    sigma_hat_unbiased = np.sqrt(np.var(data, ddof=1))

    fig, ax = plt.subplots()
    ax.hist(data, bins=30, density=True, alpha=0.6, color='blue')
    x = np.linspace(mu_hat - 3 * sigma_hat_mle, mu_hat + 3 * sigma_hat_mle, 100)
    ax.plot(x, stats.norm.pdf(x, mu_hat, sigma_hat_mle), 'r-',
            label=f"MLE: N({mu_hat:.2f}, {sigma_hat_mle:.2f})")
    ax.plot(x, stats.norm.pdf(x, mu_true, sigma_true), 'g--', label=f"真实: N({mu_true}, {sigma_true})")
    ax.set_title(f"正态分布最大似然估计 (样本量: {request.n})")
    ax.set_xlabel("样本值")
    ax.set_ylabel("密度")
    ax.legend()
    img = create_plot(fig)

    return {
        "true_values": {"mu": mu_true, "sigma": sigma_true},
        "estimates": {
            "mu": mu_hat,
            "sigma_mle": sigma_hat_mle,
            "sigma_unbiased": sigma_hat_unbiased
        },
        "errors": {
            "mu": abs(mu_hat - mu_true),
            "sigma_mle": abs(sigma_hat_mle - sigma_true),
            "sigma_unbiased": abs(sigma_hat_unbiased - sigma_true)
        },
        "image": img
    }


@app.post("/api/chapter7/efficiency")
async def estimator_efficiency(request: EstimatorEfficiencyRequest):
    try:
        num_samples = 500
        estimates = []
        crlb_values = []

        for _ in range(num_samples):
            if request.dist_type == "正态分布":
                sample = np.random.normal(5, 2, request.sample_size)
                estimates.append(np.mean(sample))
                crlb = 4 / request.sample_size
                crlb_values.append(crlb)
            elif request.dist_type == "均匀分布":
                sample = np.random.uniform(0, 10, request.sample_size)
                estimates.append(np.mean(sample))
                crlb = (10 ** 2 / 12) / request.sample_size
                crlb_values.append(crlb)
            elif request.dist_type == "泊松分布":
                sample = np.random.poisson(5, request.sample_size)
                estimates.append(np.mean(sample))
                crlb = 5 / request.sample_size
                crlb_values.append(crlb)
            else:
                raise HTTPException(status_code=400, detail=f"不支持的分布类型: {request.dist_type}")

        var_estimate = np.var(estimates)
        avg_crlb = np.mean(crlb_values)

        fig, ax = plt.subplots()
        ax.hist(estimates, bins=30, alpha=0.6, color='blue')
        ax.set_title(f"{request.dist_type} 估计量分布")
        ax.set_xlabel("估计值")
        ax.set_ylabel("频率")
        img = create_plot(fig)

        return {
            "variance": float(var_estimate),
            "crlb": float(avg_crlb),
            "is_efficient": bool(var_estimate < avg_crlb * 1.05),
            "image": img
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"服务器错误: {str(e)}")



# 将整个函数替换为:
@app.post("/api/chapter8/confidence-interval")
async def confidence_interval(request: ConfidenceIntervalRequest):
    try:
        population_mean = 50
        population_std = 10
        sample = np.random.normal(population_mean, population_std, request.sample_size)

        sample_mean = float(np.mean(sample))
        sample_std = float(np.std(sample, ddof=1))

        alpha = 1 - request.confidence_level
        t_value = float(stats.t.ppf(1 - alpha / 2, df=request.sample_size - 1))
        margin_error = t_value * sample_std / np.sqrt(request.sample_size)
        ci_lower = sample_mean - margin_error
        ci_upper = sample_mean + margin_error

        fig, ax = plt.subplots()
        ax.hist(sample, bins=30, alpha=0.6, color='blue')
        ax.axvline(x=sample_mean, color='red', linestyle='-', label=f"样本均值: {sample_mean:.2f}")
        ax.axvline(x=ci_lower, color='green', linestyle='--',
                   label=f"{request.confidence_level * 100}% CI: [{ci_lower:.2f}, {ci_upper:.2f}]")
        ax.axvline(x=ci_upper, color='green', linestyle='--')
        ax.axvline(x=population_mean, color='black', linestyle='-.', label=f"总体均值: {population_mean}")
        ax.set_title("样本数据分布")
        ax.set_xlabel("样本值")
        ax.set_ylabel("频率")
        ax.legend()
        img = create_plot(fig)

        return {
            "sample_mean": sample_mean,
            "sample_std": sample_std,
            "ci_lower": float(ci_lower),
            "ci_upper": float(ci_upper),
            "margin_error": float(margin_error),
            "contains_true_mean": bool(ci_lower <= population_mean <= ci_upper),
            "image": img
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"服务器错误: {str(e)}")




@app.post("/api/chapter8/normal-ci")
async def normal_confidence_interval(request: Dict[str, Any]):
    pop_mean = request.get('pop_mean', 0.0)
    pop_std = request.get('pop_std', 1.0)
    sample_size = request.get('sample_size', 50)
    confidence_level = request.get('confidence_level', 0.95)

    sample = np.random.normal(pop_mean, pop_std, sample_size)
    sample_mean = np.mean(sample)
    sample_std = np.std(sample, ddof=1)

    alpha = 1 - confidence_level
    t_value = stats.t.ppf(1 - alpha / 2, df=sample_size - 1)
    margin_error_mean = t_value * sample_std / np.sqrt(sample_size)
    ci_mean = [sample_mean - margin_error_mean, sample_mean + margin_error_mean]

    chi2_lower = stats.chi2.ppf(alpha / 2, df=sample_size - 1)
    chi2_upper = stats.chi2.ppf(1 - alpha / 2, df=sample_size - 1)
    ci_var = [
        (sample_size - 1) * sample_std ** 2 / chi2_upper,
        (sample_size - 1) * sample_std ** 2 / chi2_lower
    ]

    fig1, ax1 = plt.subplots()
    ax1.errorbar(x=0, y=sample_mean, yerr=margin_error_mean, fmt='bo', capsize=5, label="样本均值")
    ax1.axhline(y=pop_mean, color='r', linestyle='--', label="真实均值")
    ax1.set_title(f"均值的{confidence_level * 100}%置信区间")
    ax1.set_xlim(-0.5, 0.5)
    ax1.set_xticks([])
    ax1.legend()
    img1 = create_plot(fig1)

    fig2, ax2 = plt.subplots()
    yerr = [[sample_std ** 2 - ci_var[0]], [ci_var[1] - sample_std ** 2]]
    ax2.errorbar(x=0, y=sample_std ** 2, yerr=yerr, fmt='go', capsize=5, label="样本方差")
    ax2.axhline(y=pop_std ** 2, color='r', linestyle='--', label="真实方差")
    ax2.set_title(f"方差的{confidence_level * 100}%置信区间")
    ax2.set_xlim(-0.5, 0.5)
    ax2.set_xticks([])
    ax2.legend()
    img2 = create_plot(fig2)

    return {
        "mean_ci": ci_mean,
        "var_ci": ci_var,
        "contains_mean": ci_mean[0] <= pop_mean <= ci_mean[1],
        "contains_var": ci_var[0] <= pop_std ** 2 <= ci_var[1],
        "mean_image": img1,
        "var_image": img2
    }


@app.post("/api/chapter9/one-sample-ttest")
async def one_sample_ttest(request: TTestRequest):
    population_std = 2.0
    sample = np.random.normal(request.true_mean, population_std, request.sample_size)

    sample_mean = np.mean(sample)
    sample_std = np.std(sample, ddof=1)

    t_stat, p_value = stats.ttest_1samp(sample, request.hypothesized_mean)
    df = request.sample_size - 1

    t_critical = stats.t.ppf(1 - request.alpha / 2, df)
    reject_low = request.hypothesized_mean - t_critical * (sample_std / np.sqrt(request.sample_size))
    reject_high = request.hypothesized_mean + t_critical * (sample_std / np.sqrt(request.sample_size))

    decision = "拒绝原假设" if p_value < request.alpha else "不拒绝原假设"

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(sample, bins=30, alpha=0.6, color='blue', density=True, label="样本分布")
    x = np.linspace(min(sample), max(sample), 100)
    ax.plot(x, stats.norm.pdf(x, sample_mean, sample_std), 'r-', label="样本正态近似")
    ax.axvline(x=request.hypothesized_mean, color='green', linestyle='--',
               label=f"假设均值 μ₀={request.hypothesized_mean}")
    ax.axvline(x=request.true_mean, color='black', linestyle='-.', label=f"真实均值 μ={request.true_mean}")
    ax.axvline(x=sample_mean, color='red', linestyle='-', label=f"样本均值={sample_mean:.2f}")
    ax.axvspan(min(sample), reject_low, color='gray', alpha=0.3, label="拒绝域")
    ax.axvspan(reject_high, max(sample), color='gray', alpha=0.3)
    ax.set_title(f"单样本t检验结果 (α={request.alpha})")
    ax.set_xlabel("样本值")
    ax.set_ylabel("密度")
    ax.legend()
    img = create_plot(fig)

    return {
        "t_stat": t_stat,
        "p_value": p_value,
        "df": df,
        "t_critical": t_critical,
        "decision": decision,
        "sample_mean": sample_mean,
        "image": img
    }


@app.post("/api/chapter9/two-sample-ttest")
async def two_sample_ttest(request: TwoSampleTTestRequest):
    sample1 = np.random.normal(request.mean1, request.std1, request.sample_size1)
    sample2 = np.random.normal(request.mean2, request.std2, request.sample_size2)

    mean1_sample = np.mean(sample1)
    mean2_sample = np.mean(sample2)

    t_stat, p_value = stats.ttest_ind(sample1, sample2, equal_var=request.equal_var)
    decision = "拒绝原假设" if p_value < request.alpha else "不拒绝原假设"

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(sample1, bins=30, alpha=0.5, color='blue', density=True, label="样本1")
    ax.hist(sample2, bins=30, alpha=0.5, color='green', density=True, label="样本2")
    ax.axvline(x=mean1_sample, color='blue', linestyle='-', label=f"样本1均值={mean1_sample:.2f}")
    ax.axvline(x=mean2_sample, color='green', linestyle='-', label=f"样本2均值={mean2_sample:.2f}")
    ax.set_title("两样本分布比较")
    ax.set_xlabel("样本值")
    ax.set_ylabel("密度")
    ax.legend()
    img = create_plot(fig)

    return {
        "t_stat": t_stat,
        "p_value": p_value,
        "mean_diff": mean1_sample - mean2_sample,
        "decision": decision,
        "mean1": mean1_sample,
        "mean2": mean2_sample,
        "image": img
    }


@app.post("/api/chapter9/t-test-animation")
async def t_test_animation(request: TTestRequest):
    """t检验动图"""
    try:
        from PIL import Image

        n_frames = 30
        sample = np.random.normal(request.true_mean, 1, request.sample_size)

        fig, ax = plt.subplots(figsize=(12, 7))
        frames = []

        for frame_idx in range(1, n_frames + 1):
            current_size = max(5, int(frame_idx * request.sample_size / n_frames))
            current_sample = sample[:current_size]

            sample_mean = np.mean(current_sample)
            sample_std = np.std(current_sample, ddof=1)

            t_stat = (sample_mean - request.hypothesized_mean) / (sample_std / np.sqrt(current_size))

            x = np.linspace(-4, 4, 300)
            df = current_size - 1
            t_dist = stats.t.pdf(x, df)

            ax.clear()
            ax.plot(x, t_dist, 'b-', linewidth=2.5, label=f't分布 (df={df})')

            ax.fill_between(x[x <= -abs(t_stat)], t_dist[x <= -abs(t_stat)],
                            color='red', alpha=0.3, label='拒绝域')
            ax.fill_between(x[x >= abs(t_stat)], t_dist[x >= abs(t_stat)],
                            color='red', alpha=0.3)

            ax.axvline(x=t_stat, color='green', linestyle='--',
                       linewidth=2.5, label=f't统计量 = {t_stat:.3f}')
            ax.axvline(x=request.hypothesized_mean, color='orange',
                       linestyle=':', linewidth=2, label=f'H₀: μ={request.hypothesized_mean}')

            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))

            ax.set_title(f'单样本t检验动态演示\n样本量: {current_size}, t统计量: {t_stat:.3f}, '
                         f'p值: {p_value:.4f}\n{"拒绝H₀" if p_value < request.alpha else "不拒绝H₀"} '
                         f'(α={request.alpha})',
                         fontsize=12, fontweight='bold')
            ax.set_xlabel('t值', fontsize=12)
            ax.set_ylabel('概率密度', fontsize=12)
            ax.legend(loc='best', fontsize=9)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.set_ylim(0, max(t_dist) * 1.1)

            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight', dpi=80)
            buf.seek(0)
            img = Image.open(buf)
            frames.append(img.copy())
            buf.close()

        plt.close(fig)

        if len(frames) > 1:
            gif_image = create_gif_animation(frames, duration=150)
        else:
            gif_image = None

        sample_mean_full = np.mean(sample)
        sample_std_full = np.std(sample, ddof=1)
        t_stat_full = (sample_mean_full - request.hypothesized_mean) / (sample_std_full / np.sqrt(request.sample_size))
        p_value_full = 2 * (1 - stats.t.cdf(abs(t_stat_full), request.sample_size - 1))

        return {
            "sample_mean": sample_mean_full,
            "sample_std": sample_std_full,
            "t_statistic": t_stat_full,
            "p_value": p_value_full,
            "reject_null": p_value_full < request.alpha,
            "animation": gif_image,
            "total_frames": len(frames)
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"生成动图失败: {str(e)}")


@app.post("/api/code-runner/execute")
async def execute_user_code(request: CodeExecutionRequest):
    """执行用户提交的 Python 代码并返回生成的图表"""
    import time
    import sys
    from io import StringIO

    start_time = time.time()

    try:
        old_stdout = sys.stdout
        redirected_output = sys.stdout = StringIO()

        local_vars = {}
        global_vars = {
            'np': np,
            'plt': plt,
            'pd': pd,
            'stats': stats,
            '__name__': '__main__'
        }

        exec(request.code, global_vars, local_vars)

        sys.stdout = old_stdout
        output = redirected_output.getvalue()

        figs = [manager.canvas.figure for manager in plt._pylab_helpers.Gcf.get_all_fig_managers()]

        if not figs:
            execution_time = time.time() - start_time
            return {
                "success": True,
                "message": "代码执行成功，但未生成图表",
                "output": output,
                "execution_time": execution_time
            }

        last_fig = figs[-1]
        img_base64 = create_plot(last_fig)

        plt.close('all')

        execution_time = time.time() - start_time

        return {
            "success": True,
            "image": img_base64,
            "output": output,
            "execution_time": execution_time
        }

    except Exception as e:
        execution_time = time.time() - start_time
        traceback.print_exc()
        raise HTTPException(
            status_code=400,
            detail=f"代码执行错误:\n{str(e)}"
        )


@app.post("/api/chapter9/t-test")
async def t_test_alias(request: TTestRequest):
    """单样本t检验的别名路由"""
    return await one_sample_ttest(request)


@app.post("/api/chapter9/two-sample-t-test")
async def two_sample_t_test_alias(request: TwoSampleTTestRequest):
    """双样本t检验的别名路由"""
    return await two_sample_ttest(request)



@app.post("/api/ai/chat")
async def ai_chat(request: AIMessage):
    try:
        api_key = os.getenv("OPENAI_API_KEY", "")
        ai_provider = os.getenv("AI_PROVIDER", "openai").lower()

        if not api_key or api_key == "your_api_key_here":
            return {
                "response": "🤖 AI 教学助手\n\n"
                            "当前未配置 API 密钥，AI 功能暂不可用。\n\n"
                            "💡 配置方法：\n\n"
                            "【使用通义千问（推荐）】\n"
                            "1. 访问 https://dashscope.console.aliyun.com\n"
                            "2. 注册并创建 API-KEY\n"
                            "3. 在 .env 文件中设置：\n"
                            "   AI_PROVIDER=dashscope\n"
                            "   OPENAI_API_KEY=你的通义千问API-Key\n\n"
                            "📚 不过没关系！你仍然可以使用平台的所有概率实验功能！",
                "success": False
            }

        system_prompt = """你是一个专业的概率论与数理统计教学助手。你的任务是帮助学生理解概率统计概念、解释实验结果、回答相关问题。请用简洁易懂的语言回答，必要时可以给出数学公式和示例。"""

        user_message = request.message
        if request.context:
            user_message = f"当前学习内容：{request.context}\n\n问题：{user_message}"

        if ai_provider == "dashscope":
            if not DASHSCOPE_AVAILABLE:
                return {
                    "response": "dashscope库未安装。请运行 pip install dashscope 安装通义千问SDK。",
                    "success": False
                }
            return await call_dashscope(system_prompt, user_message, api_key)
        else:
            if not OPENAI_AVAILABLE:
                return {
                    "response": "OpenAI库未安装。请运行 pip install openai 安装。",
                    "success": False
                }
            return await call_openai(system_prompt, user_message, api_key)

    except Exception as e:
        return {
            "response": f"AI服务出错：{str(e)}",
            "success": False
        }


async def call_openai(system_prompt, user_message, api_key):
    """调用 OpenAI API"""
    try:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]

        client = OpenAI(
            api_key=api_key,
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        )

        import asyncio
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=800,
                temperature=0.7
            )
        )

        return {
            "response": response.choices[0].message.content,
            "success": True
        }
    except Exception as e:
        error_msg = str(e)
        if "authentication" in error_msg.lower() or "api key" in error_msg.lower():
            return {
                "response": "API密钥验证失败。请检查OPENAI_API_KEY是否正确配置。",
                "success": False
            }
        elif "rate limit" in error_msg.lower():
            return {
                "response": "API调用频率超限。请稍后再试。",
                "success": False
            }
        else:
            return {
                "response": f"OpenAI服务出错：{error_msg}",
                "success": False
            }


async def call_dashscope(system_prompt, user_message, api_key):
    """调用阿里云通义千问 API"""
    try:
        from http import HTTPStatus

        dashscope.api_key = api_key

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]

        import asyncio
        loop = asyncio.get_event_loop()

        def call_api():
            response = dashscope.Generation.call(
                model=dashscope.Generation.Models.qwen_turbo,
                messages=messages,
                result_format='message'
            )
            return response

        response = await loop.run_in_executor(None, call_api)

        if response.status_code == HTTPStatus.OK:
            return {
                "response": response.output.choices[0].message.content,
                "success": True
            }
        else:
            return {
                "response": f"通义千问API调用失败：{response.code} - {response.message}",
                "success": False
            }

    except Exception as e:
        error_msg = str(e)
        if "ModuleNotFoundError" in error_msg or "No module named 'dashscope'" in error_msg:
            return {
                "response": "dashscope库未安装。请运行 pip install dashscope 安装。",
                "success": False
            }
        else:
            return {
                "response": f"通义千问服务出错：{error_msg}",
                "success": False
            }


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("APP_HOST", "0.0.0.0")
    port = int(os.getenv("APP_PORT", "8000"))
    debug = os.getenv("DEBUG", "false").lower() == "true"
    uvicorn.run(app, host=host, port=port, reload=debug)
