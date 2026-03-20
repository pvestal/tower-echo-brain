from fastapi import APIRouter

router = APIRouter(tags=["search"])

@router.get("/search")
async def search_disabled():
    return {"status": "disabled", "message": "Search router temporarily disabled due to database password issue"}
